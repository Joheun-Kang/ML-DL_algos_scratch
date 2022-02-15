# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import platform
print("Python version: ",platform.python_version())

import pandas as pd
import os
import re
import json
import pickle
import sys
import signal
import traceback
import pickle
import optparse
from csv import reader
import flask
import sklearn

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

# Set environment to test on
#test_type = 'docker'
test_type = 'local'

# Set directory and import helper functions based on environment; docker run creates "opt/ml/" directory
if test_type == 'docker':
    import helper_functions_nlp as hf
    prefix = '/opt/ml/'
    model_path = '../ml/model/demofinal.light.clf/'
else:
    from container.decision_trees import helper_functions_nlp
    prefix = 'container/local_test/test_dir/'
    model_path = os.path.join(prefix, 'model')


print("Cur dir: ",os.getcwd())

# Model name (model will be loaded in format <model_name>.pkl)
model_name = 'demofinal.light.clf'

# Max Sentence number to infer
sentence_threshold = 500

# need this 
class FeatureGroup(object):
    def __init__(self, name, homedir='./'):
        self.name = name
        self.homedir = homedir

    def fname(self, f) : return '%s_%s'%(self.name, f)

    def add(self, features, feature, value): 
        features[self.fname(feature)] = float(value)
        return features
    
    def inc(self, features, feature, value):
        fnm = self.fname(feature)
        value = float(value)
        if fnm in features:
            features[fnm] += value
        else:
            self.add(features, feature, value) 
        return features

    def norm_by(self, features, d, exclude=[]): 
        exclude = [self.fname(f) for f in exclude]
        d = float(d)
        return {f : v/d for f,v in features.iteritems() if f not in exclude}

    def feature_string(self, features): 
        return ' '.join(['%s:%s'%(f,v) for f,v in features.iteritems()])



class NgramFeatures(FeatureGroup):
    def __init__(self, name, homedir='./', N=3):
        super(NgramFeatures, self).__init__(name, homedir)
        self.N = N

    def extract(self, doc):
        from nltk.util import ngrams
        features = {}
        for sent in doc:
            words = sent['tokens']
        for n in range(1,self.N):
            for ngm in ngrams(words,n):
                features = self.add(features, ' '.join([w.lower() for w in ngm]), 1)
        return features

class CaseFeatures(FeatureGroup):
    def extract(self, doc):
        features = {}
        for sent in doc:
            caps = sum([1 if (w.isupper() and not ((w == 'I'))) else 0 for w in sent['tokens']])
            features = self.inc(features, 'Number of capitalized words', caps)
            all_lower = sum([0 if w.islower() else 1 for w in sent['tokens'] if w.isalpha()]) > 0 
            features = self.inc(features, 'All lowercase sentence', all_lower)
            features = self.inc(features, 'Lowercase initial sentence', 1 if sent['tokens'][0].islower() else 0)

            for punct in ['!', '?', '...'] : 
                for w in sent['tokens']:
                    if punct in w: 
                        features = self.inc(features, 'Number of %s per sentence'%punct, 1)

        return features


class ReadabilityFeatures(FeatureGroup) :
    def __init__(self, name, homedir='./'):
        from nltk.corpus import cmudict
        #from nltk import cmudict
        super(ReadabilityFeatures, self).__init__(name, homedir)
        self.d = cmudict.dict()

  #from https://groups.google.com/forum/#!topic/nltk-users/mCOh_u7V8_I
    def _nsyl(self, word): 
        import curses
        from curses.ascii import isdigit
        word = word.lower()
        if word in self.d : 
            return min([len(list(y for y in x if y[-1].isdigit())) for x in self.d[word.lower()]])
        else : return 0

    def _FK(self, toks): 
        words =0.
        sents = 1.
        syllables = 0.
        for w in toks:
            words += 1
            syllables += self._nsyl(w)
        if words > 0 and sents > 0 : 
              return (0.39 * (words/sents)) + (11.8 * (syllables/words)) - 15.59
        return 0

    def extract(self, doc):
        features = {}
        alltoks = []
        for sent in doc: 
            alltoks += sent['tokens']
            features = self.inc(features, 'length in words', len(sent['tokens']))
            features = self.inc(features, 'length in characters', len(' '.join(sent['tokens'])))
        features = self.add(features, 'FK score', self._FK(alltoks))
        return features

class W2VFeatures(FeatureGroup):
    def __init__(self, name, homedir='./'):
        #from gensim.models import word2vec, doc2vec #original
        import gensim.models.keyedvectors as word2vec # new 

        super(W2VFeatures, self).__init__(name, homedir)
        #Load word2vec pretrained vectors
        sys.stderr.write("Loading w2v...")
        
        self.w2v = word2vec.KeyedVectors.load_word2vec_format('/Users/joheunkang/Desktop/Lavender_fall/NLP_lavender_fall/formality_deploy/help/GoogleNews-vectors-negative300.bin', binary = True) #this works
        sys.stderr.write("done\n")

    def extract(self, doc): 
        import numpy as np
        features = {}
        v = None
        d1 = None
        total = 0.
        for sent in doc:
            d1 = doc
            for w in sent['tokens'] : 
                try :
                    wv = np.array(self.w2v[w.lower()])
                    if (max(wv) < float('inf')) and (min(wv) > -float('inf')) : 
                        if v is None : v = wv
                        else : v += wv
                        total += 1
                except KeyError : 
                      continue
            if v is not None :
                v = v / total
                for i,n in enumerate(v):
                    if (n == float('inf')) : n = sys.float_info.max
                    if (n == -float('inf')) : n = -sys.float_info.max
                    features = self.add(features, 'w2v-%d'%i, n)
            else : 
                features = self.add(features, 'w2v-NA', 1)
        return features


class Featurizer: 
    def __init__(self, parsecachepath='ref/featurizer.parsecache', use='all', homedir='./', reload_parses=True, cache_dump_freq=200000, preproc='nltk'):
        self.homedir = homedir
        
        self.light_feature_names = ['ngram', 'case', 'readability', 'w2v']
        self.feature_names = ['pos', 'subjectivity', 'lexical', 'ngram', 'case', 'entity', 'constituency', 'dependency', 'readability', 'w2v']
    
        sys.stderr.write("Initializing featurizer\n")
    
        if use == 'light' : use = set(self.light_feature_names)
        elif use == 'all' : use = set(self.feature_names)
        else : use = set(use.split(','))
        
        #print('what is use',use)
        
        #if using parse or entity features, override preproc option and force to use stanford
        if 'constituency' in use: 
            if not(preproc == 'stanford'): 
                sys.stderr.write('Warning: using constituency features requires use of StanfordPreprocessor. This might cause errors if the required dependencies are not installed\n')
                preproc = 'stanford' 
        if 'dependency' in use: 
            if not(preproc == 'stanford'): 
                sys.stderr.write('Warning: using dependency features requires use of StanfordPreprocessor. This might cause errors if the required dependencies are not installed\n')
                preproc = 'stanford' 
        if 'entity' in use: 
              if not(preproc == 'stanford'):
                    sys.stderr.write('Warning: using entity features requires use of StanfordPreprocessor. This might cause errors if the required dependencies are not installed\n')
                    preproc = 'stanford' 

        self.use_features = self._get_feature_to_use(use)

        #Initialize preprocessor; Stanford parser required only if using dependency, constituency, or entity features

        self.preprocessor = NLTKPreprocessor()

        #Load existing parse cache, or create a new one
        self.parsecache = {}
        self.parsecachepath = parsecachepath+'_'+preproc+'.pkl'
        if not(os.path.exists(self.parsecachepath)) : 
            self.parsecache = {}
        else :
            sys.stderr.write('Loading parse cache...')
            if reload_parses :
                self.parsecache = pickle.load(open(self.parsecachepath,"rb")) #rb was not there
            else : 
                self.parsecache = {}
            sys.stderr.write('done\n')

        #Sundry other initializations
        self.URL = re.compile("(www|(https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")
        self.EMAIL = re.compile("[^@]+@[^@]+\.[^@]+")
        self.new_parses = 0
        self.cache_dump_freq = cache_dump_freq
    
    def _get_feature_to_use(self, use): 
        use_features = []
        #print('what is use in _get_feature_to_use',use)
        #light
        if 'case' in use: use_features.append(CaseFeatures('case', homedir='./'))

        if 'ngram' in use:
            use_features.append(NgramFeatures('ngram', homedir=self.homedir))

        if 'readability' in use:
            use_features.append(ReadabilityFeatures('readability', homedir=self.homedir)) # issue

        # if 'w2v' in use: 
        #   use_features.append(W2VFeatures('w2v', homedir=self.homedir))

        return use_features


    def dump_cache(self) : 
        """Save any unsaved parses to cache"""
        if self.new_parses > 0 :
            sys.stderr.write('Saving parses to cache...\n')
            pickle.dump(self.parsecache, open(self.parsecachepath, 'wb')) # wb changed
            sys.stderr.write('Save complete.\n')
            self.new_parses = 0
    def close(self) : 
        """Close the Featurizer and save parse cache if necessary"""
        self.dump_cache()

    def _replace_urls(self, s):
        """Replace urls and emails with special token"""
        ret = '' 
        for w in s.split() :
            if re.match(self.URL, w) : ret += '_url_ '
            elif re.match(self.EMAIL, w) : ret += '_email_ '
            else : ret += '%s '%w
        return ret.strip()

    def featurize(self, s, sid=None, use=None):
        """Extract all of the features for sentence

        s -- the sentence (or any text blob, but will work best if it is just a single sentence
        sid -- an identifier for the sentence, used to key into the cache
        """
        if use is None : use_features = self.use_features
        else : use_features = self._get_feature_to_use(use)

        if sid is None : sid = s
        s = self._replace_urls(s)
        if '%s'%sid in self.parsecache :
            sent = self.parsecache['%s'%sid]
        else : 
            sent = self.preprocessor.parse(s)['sentences']
            self.parsecache['%s'%sid] = sent
            self.new_parses += 1
            if self.new_parses == self.cache_dump_freq: 
                self.dump_cache()
        features = {}
        for feats in use_features:
            f = feats.extract(sent)
            features.update(f)
        return features


def feature_string(d): 
    return ' '.join(['%s:%s'%(f.replace(' ', '_'), v) for f,v in d.iteritems()])

def score(sent,ftzr,dv,clf): 
    # we need input sentence 
    #print('what is type of sentence here',type(sent))
    #print('this is the input sentence', sent)
    xdict = ftzr.featurize(sent)
    x = dv.transform(xdict)
    fstr = feature_string(xdict) if dump == True else ''
    return clf.predict(x)[0], fstr

def score_doc(sentences, ftzr, dv, clf): 
    #print('this is SENTENCES',sentences)
    scores = []
    fstrs = []
    for sent in sentences : 
        s, f = score(' '.join(sent['tokens']), ftzr, dv, clf)
        scores.append(s)
        fstrs.append(f)
    avg = sum(scores)/len(scores)
    return avg, scores, fstrs


class NLTKPreprocessor(object): 
    def __init__(self, homedir='./'):
        from nltk.stem.wordnet import WordNetLemmatizer
        self.ltzr = WordNetLemmatizer()

    def _get_wordnet_pos(self, tag):
        from nltk.corpus import wordnet 
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN #this is the default when POS is not known
 
    def parse(self, document): 
        from nltk.tokenize import word_tokenize, sent_tokenize
        from nltk.tag import pos_tag
        data = {'sentences': []}
        for sent in sent_tokenize(document):
            tokens = [w for w in word_tokenize(sent)]
            postags = [t for w,t in pos_tag(tokens)]
            lemmas = [self.ltzr.lemmatize(w,self._get_wordnet_pos(t)) for w,t in zip(tokens, postags)]
            data['sentences'].append({'tokens' : tokens, 'lemmas' : lemmas, 'pos': postags})
        return data



# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
               
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            
            if test_type == 'docker':
                # AWS unzips model from S3 location. Customize loading paths accordingly
                print("Model path files: ", os.listdir('../ml/model/'))
                print("Model path unzipped files: ", os.listdir(model_path))

            with open(os.path.join(model_path, model_name + '.clf'), 'rb') as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        
        # Set number of workers to 1 to avoid issues on Ubuntu (in case training was run parallely with n_jobs > 1)
        clf['model'].n_jobs = 1
        
        return clf.predict_proba(input)

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions to Json
    """
   
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data

        s = StringIO(data.decode('utf-8'))
        
        data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    
    # Do not infer if sentences cross threshold
    if data.shape[0] > sentence_threshold:
        return flask.Response(response='Input text data is too long, please enter sentences fewer than {}'.format(sentence_threshold), status=413, mimetype='text/plain')
    
    try:
        # Print for debugging, see output in Endpoint Cloudwatch
        print("Incoming data: ", data)
        print("Incoming data datatypes: ", data.dtypes)

        # Convert data to string, since dealing with text inputs
        data = data.astype(str)

        # Name column
        data.columns = ['input_text']

        # Clean text field using same steps from model training
        data = hf.clean_text_basic(data, text_to_clean = data.columns[0])

        # Preprocess text field using same steps from model training
        data = hf.preprocess_cleaned_text(data, text_to_preprocess = "cleaned_text", lemmatize_flag = True)

        print("Processed data: ", data.cleaned_processed)

        # Do the prediction
        predictions = ScoringService.predict(data["cleaned_processed"].values)

        print("Predictions: ", predictions)

        # Convert from numpy back to CSV
        out = StringIO()

        # Categories from model training
        categories = ['anger', 'fear', 'joy', 'sadness']

        # Assign predictions in new field
        data['predictions'] = [{_cat: _pred for _cat, _pred in zip(categories, _pred)} for _pred in predictions]

        # Convert predictions to CSV
        #data[['input_text', 'predictions']].to_csv(out, header=False, index=False)
        #result = out.getvalue()

        # Convert predictions to dictionary for easy conversion to json
        result = data[['input_text', 'predictions']].to_dict('index')

        # Return json predictions
        return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')

        # Return CSV predictions
        #return flask.Response(response=result, status=200, mimetype='text/csv')
        
    except Exception as e:
        print("Error in transform script: ", e)
        return flask.Response(response="Error in transform script: {}".format(e), status=417, mimetype='text/plain')
    
