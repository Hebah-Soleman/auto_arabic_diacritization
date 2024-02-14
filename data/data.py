import re
import os
from collections import Counter
import json
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from arabic_reshaper import arabic_reshaper
from bidi.algorithm import get_display


class DataVisualizer:
  def __init__(self, text,font_file_path='data/constants/arial.ttf'):
    self.text = text
    self.font_file_path = font_file_path

  def most_common_words_hist(self, top_n = 20):
    word_counts = Counter(self.text.split())
    top_words = [word[0] for word in word_counts.most_common(top_n)]
    word_freq = [word_counts[word] for word in top_words]
    plt.figure(figsize=(10, 6))
    plt.barh(top_words, word_freq, color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title('Top {} Most Common Words'.format(top_n))
    plt.show()

  def n_gram(self):
    #text = self.text.split()
    cv = CountVectorizer(ngram_range=(3,3))
    bigrams = cv.fit_transform(self.text.split('\n'))
    count_values = bigrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ["frequency", "ngram"]
    sns.barplot(x=ngram_freq['frequency'][:10], y=ngram_freq['ngram'][:10])
    plt.title('Top 10 Most Frequently Occuring Bigrams')
    plt.show()

  def most_common_word_df(self):
    #text = self.text.split()
    top_common = Counter([vocab for vocab in self.text.split(' ')])
    top_common = pd.DataFrame(top_common.most_common(20))
    top_common.columns = ['Common_words','Frequency']
    return top_common.style.background_gradient(cmap='Blues')

  def vocab_frequency(self):
    #text = self.text.split()
    vocab = Counter([item for sublist in self.text for item in sublist])
    top_common = pd.DataFrame(vocab.most_common(50))
    top_common.columns = ['Common_vocab','Frequency']
    return top_common.style.background_gradient(cmap='Blues')

  def word_cloud(self):
    reshaped_text = arabic_reshaper.reshape(self.text)
    arabic_text = get_display(reshaped_text)
    wordcloud = WordCloud(width=800,
                          height=400,
                          background_color='white',
                          font_path = self.font_file_path).generate(arabic_text)
    plt.figure(figsize=(10, 6), facecolor=None)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    
        
class DataPreprocessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.text = self._read_text()
        self.DIACRITICS_LIST = ['َ', 'ً', 'ِ', 'ٍ', 'ُ', 'ٌ', 'ْ', 'ّ']
    def _read_text(self):
        text = ""
        folder_path = self.folder_path+'/dataset'
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    content = file.read()
                    non_empty_lines = [line.strip() for line in content.split('\n') if line.strip()]
                    non_empty_lines = '\n'.join(non_empty_lines)
                    text += non_empty_lines + "\n"
        return text

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def remove_special_characters(self, text):
        pattern = r'[a-zA-Z0-9()-\[\],.;!؟?\\\/{}،؛]'
        text = re.sub(pattern, '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(" هـ ", "")
        text = text.replace(" ا ", "")
        text = text.replace(" ص ", "")
        text = text.replace(" م ", "")
        text = text.replace( 'ـ', "")
        text = text.replace(" أ ", "")
        text = text.replace('\u200d', "")
        return text

    def remove_diacritics(self, data_raw):
        return data_raw.translate(str.maketrans('', '', ''.join(self.DIACRITICS_LIST)))

    def tokenize_phrases(self, data_raw):
        data_new = list()
        line = data_raw.replace('.', '.\n')
        line = line.replace(',', ',\n')
        line = line.replace('،', '،\n')
        line = line.replace(':', ':\n')
        line = line.replace(';', ';\n')
        line = line.replace('؛', '؛\n')
        line = line.replace('(', '\n(')
        line = line.replace(')', ')\n')
        line = line.replace('[', '\n[')
        line = line.replace(']', ']\n')
        line = line.replace('{', '\n{')
        line = line.replace('}', '}\n')
        line = line.replace('«', '\n«')
        line = line.replace('»', '»\n')

        for sub_line in line.split('\n'):
            # do nothing if line is empty
            if len(self.remove_diacritics(sub_line).strip()) == 0:
                continue
            # append line to list if line, without diacritics, is shorter than 500 characters
            if len(self.remove_diacritics(sub_line).strip()) > 0 and len(self.remove_diacritics(sub_line).strip()) <= 100:
                data_new.append(sub_line.strip())
            # split line if its longer than 500 characters
            else:
                sub_line = sub_line.split()
                tmp_line = ''
                for word in sub_line:
                    # append line without current word if new word will make it exceed 500 characters and start new line
                    if len(self.remove_diacritics(tmp_line).strip()) + len(self.remove_diacritics(word).strip()) + 1 > 100:
                        if len(self.remove_diacritics(tmp_line).strip()) > 0:
                            data_new.append(tmp_line.strip())
                        tmp_line = word
                    else:
                        # set new line to current word if line is still empty
                        if tmp_line == '':
                            tmp_line = word
                        # add whitespace and word to line if line is not empty but shorter than 500 characters
                        else:
                            tmp_line += ' '
                            tmp_line += word
                if len(self.remove_diacritics(tmp_line).strip()) > 0:
                    data_new.append(tmp_line.strip())
        phrases = [self.remove_special_characters(sentence) for sentence in data_new]
        return phrases

    def tokenize_words(self, text):
        text = self.remove_special_characters(text)
        return text.split()

    def tokenize_Charecters(self):
        Charecters_Tokens = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2,}
        last_index = max(Charecters_Tokens.values())
        Charecters_Tokens.update({char: idx+last_index+1 for idx, char in enumerate(set(''.join(self.phrases)))})
        return Charecters_Tokens

    def tokenize_Charecters_to_file(self):
      output_file_name = 'constants/CHARACTER_TOKENS.json'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w') as json_file:
          json.dump(self.tokenize_Charecters(), json_file)

    def tokenize_diacritization(self):
        diacritization_chars = 'ًٌٍَُِّْ'
        diacritization_tokens = {}
        diacritization_tokens.update({char: idx  for idx, char in enumerate(diacritization_chars)})
        return diacritization_tokens

    def tokenize_diacritization_to_file(self):
      output_file_name = 'constants/DIACRITIZATION.json'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w') as json_file:
          json.dump(self.tokenize_diacritization(), json_file)

    def tokenize_classes(self):
        diacritization_chars = 'ًٌٍَُِّْ'
        diacritization_tokens = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '': 3}
        last_index = max(diacritization_tokens.values())
        diacritization_tokens.update({char: idx + last_index +1 for idx, char in enumerate(diacritization_chars)})
        return diacritization_tokens

    def tokenize_calsses_to_file(self):
      output_file_name = 'constants/CLASSES.json'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w') as json_file:
          json.dump(self.tokenize_classes(), json_file)

    def tokenize_rev_classes(self):
        diacritization_tokens = {0:'<PAD>', 1: '<SOS>', 2:'<EOS>', 3:'', 4:'َ', 5 : 'ُ', 6: 'ِ', 7: 'ّ', 8: 'ْ', 9: 'ً', 11: 'ٌ', 12:'ٍ'}
        return diacritization_tokens

    def tokenize_rev_calsses_to_file(self):
      output_file_name = 'constants/REV_CLASSES.json'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w') as json_file:
          json.dump(self.tokenize_rev_classes(), json_file)

    def tokenize_arabic_charecters(self):
        arabic_char = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي'
        diacritization_tokens = {}
        diacritization_tokens.update({char: idx for idx, char in enumerate(arabic_char)})
        return diacritization_tokens

    def tokenize_arabic_charecters_to_file(self):
      output_file_name = 'constants/ARABIC_CHARACTERS.json'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w') as json_file:
          json.dump(self.tokenize_arabic_charecters(), json_file)

    def words_to_file(self):
      output_file_name = 'dataset/clean_words.txt'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w', encoding='utf-8') as output_file:
          for line in self.words:
              output_file.write(line + '\n')

    def phrases_to_file(self):
      output_file_name = 'dataset/clean_data.txt'
      output_file_path = os.path.join(self.folder_path, output_file_name)
      with open(output_file_path, 'w', encoding='utf-8') as output_file:
          for line in self.phrases:
            if line != ' ' and line != '  ':
              output_file.write(line + '\n')

    def display_words(self, n=10):
        print("Individual Words:")
        print(self.words[:n])

    def display_phrases(self, n=10):
        print("\nPhrases:")
        print(self.phrases[:n])

    def preprocess(self):
        self.text_no_html = self.remove_html_tags(self.text)
        self.words = self.tokenize_words(self.text_no_html)
        self.phrases = self.tokenize_phrases(self.text_no_html)
        self.tokenize_Charecters_to_file()
        self.tokenize_diacritization_to_file()
        self.tokenize_calsses_to_file()
        self.tokenize_arabic_charecters_to_file()
        self.tokenize_rev_calsses_to_file()
        self.phrases_to_file()
        self.words_to_file()