import numpy as np
import pandas as pd
from nmf import NMF
from sklearn.decomposition import NMF as sk_nmf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import pdb
from dictionaries import blm_dict
from nltk.stem.wordnet  import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

def vectorize(content, max_features=5000):
    from sklearn.feature_extraction import text
    my_additional_stop_words = ['grouse', 'sagebrush', 'salmon', 'organ', 'idaho', \
                                'bristol', 'rio', 'grande', 'gabriel', 'ha', 'steppe', \
                                'sage', 'icl', '', 'bear', 'ear', 'staircase', 'berryessa',\
                                'elliott', 'fish', 'klamath', 'colorado', 'california', \
                                'wy', 'oregon', '000', 'likely', 'ute', 'wa', \
                                'thompson', '7th', '50th', '2b', 'doesn', '100', \
                                'ut', '350', '60', '2015', 'includes', 'le', '2014',\
                                'utah', 'pronghorn', 'million', 'billion', 'nene', \
                                'means', 'mean', 'arabella', '12', '2', '18', '2018', '2017', \
                                '8', '10', '5', '60%', '10%', '90%', '20%', '21%', '2016', \
                                '2019', '2020', '1', '3-4', '12-15', '2014', 'FY17', '50%', '2', '0', \
                                'conservation', 'foundation', 'organization', 'grantee']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    countvect = TfidfVectorizer(stop_words=stop_words, max_features=max_features, \
                                ngram_range=(1,1))
    countvect.fit(content)
    features = countvect.get_feature_names()
    return countvect.transform(content).toarray(), np.array(features)

def common_words(H, features, num_words=10):
    #pdb.set_trace()
    # weights = np.sort(H, axis=1)[:, -num_words:]
    index = np.argsort(H, axis=1)[:,-num_words:]
    index = index[::-1]
    return features[index]

def print_topics(cw):
    for e, topic in enumerate(cw):
        print ("The common words for topic {} are: {} ".format(e+1, topic))

def clean_data(list_of_strings):
    #Removes special characters
    remove_chars = [re.sub('[^A-Za-z ]', '', string) for string in list_of_strings]
    #Removes extra spaces
    clean_strings = [re.sub('\s+', ' ', s) for s in remove_chars]
    return clean_strings

def combine_words(text, dictionary):
    '''
    Takes in text that has already been lowercased but not stemmed or lematized
    Also takes in a custom dictionary for the texts

    Combines words that should be analyzed together eg 'national monunments'
    Rejoins the text in the list so it can be vectorized

    Returns a pandas series
    '''
    temp_list = []
    text_list = text.split()
    text_list = [word.replace('diversity', 'diverse') for word in text_list]
    for e, word in enumerate(text_list[0:-2]):
        next_word = text_list[e+1]
        try:
            for value in dictionary[word]:
                if value in next_word:
                    text_list.append(word+'_'+next_word)
                    temp_list.append(word)
                    temp_list.append(next_word)
        except KeyError:
            pass
    for w in temp_list:
        text_list.remove(w)
    return text_list

def lem(df_list):
    '''
    Takes in a list of words and returns list with lematized words
    '''
    lem = WordNetLemmatizer()
    lem_list = [lem.lemmatize(word) for word in df_list]
    return lem_list

def preprocess(data, column):
    text = data[column]
    lower_text = text.apply(lambda x: x.lower())
    combined = lower_text.apply(lambda x: combine_words(x, blm_dict))
    lemmatized_df = combined.apply(lambda x: lem(x))
    content = lemmatized_df.apply(lambda x: ' '.join(x))
    content = content.values
    vector, features = vectorize(content)
    return vector, features

def run_model(vector, features, k, max_iter):
    model = NMF(k, max_iter)
    W, H = model.fit_transform(vector)
    # print('Cost: ', model.cost(vector))
    cw = common_words(H, features, num_words=10)
    print ('Topics in {} with {} iterations '.format(column, max_iter))
    print_topics(cw)
    return vector, features

def optimize(data, column_list, k_list, max_iter_list):
    for col in column_list:
        for k in k_list:
            for max_iter in max_iter_list:
                vector, features = run_model(data, col, k, max_iter)
    return vector, features

def sentence_score(score_dict, text):
    sent_dict = defaultdict(int)
    text_list = text.split('.')
    for e, s in enumerate(text_list):
        score = 0
        temp_list = s.split(' ')
        for w in temp_list:
            w = w.lower()
            try:
                score += score_dict[w][0]
            except KeyError:
                continue
        sent_dict['sent{}'.format(e)]=score
    return sent_dict

def create_dfs(vector, features, data):
    ind = np.argsort(-vector, axis=1)
    top_words = features[ind][:,:15]
    top_words_df = pd.DataFrame(top_words, index=data['grantee'])
    weights = np.sort(-vector, axis=1)
    weights = weights*-1
    weights = weights[:,:15]
    weights_df = pd.DataFrame(weights, index=data['grantee'])
    top_words_df = top_words_df.where(weights_df>0)
    return top_words_df, weights_df

def overall_summary(vector, features):
    word_sum = np.sum(vector, axis=0)
    word_ind = np.argsort(-word_sum)
    overall_top = features[word_ind][:15]
    word_weight = np.sort(-word_sum)[:15]
    word_weight = word_weight*-1
    diversity_overall = pd.DataFrame(word_weight, index=overall_top)
    return diversity_overall

if __name__ == '__main__':
    column_list = ['diversity_means', 'diversity_importance', 'acted', 'support_hiring', \
                   'support_programming', 'other_advice']
    data = pd.read_csv('/Users/meghan/H-WestRef/diversity_open.csv', nrows=41)
    data = data.fillna(value='')
    columns = data.columns
    cols = [col.lower().replace(' ', '_') for col in columns]
    data.columns = cols
    for col in column_list:
        vector, features = preprocess(data, col)
        top_words_df, weights_df = create_dfs(vector, features, data)
        top_words_df.to_csv('/Users/meghan/H-WestRef/diversity_results/{}_top_words.csv'.format(col))
        weights_df.to_csv('/Users/meghan/H-WestRef/diversity_results/{}_weights.csv'.format(col))
        diversity_overall = overall_summary(vector, features)
        diversity_overall.to_csv('/Users/meghan/H-WestRef/diversity_results/{}_summary.csv'.format(col))




    '''
    Extra un-used code
    # data['enduring_combined'] = data['enduring_conservation_factors'] + data['enduring_conservation_overview']
    # k_list = [1,5]
    # max_iter_list = [100]
    # vector, features = optimize(data, column_list, k_list, max_iter_list)
    # sum_array = np.sum(vector, axis=0)
    # sum_array = sum_array.reshape(1, 348)
    # score_df = pd.DataFrame(sum_array, columns=list(features))
    # data['new'] = data['enduring_conservation_factors'].apply(lambda x: sentence_score(score_dict, x))

    # ind = np.argsort(-vector, axis=1)
    # freq_words = features[ind][:, :15]
    '''
