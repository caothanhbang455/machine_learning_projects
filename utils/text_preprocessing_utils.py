import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import UnicodeDammit
import re
from collections import Counter
from wordcloud import WordCloud
import warnings
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings('ignore')

try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("spaCy English model not found. Please install it with:")
    print("    python -m spacy download en_core_web_sm")
    raise


def load_bbc_dataset(data_path='Data/bbc/', verbose=True):
    if verbose:
        print("Loading BBC News Dataset...")
        print("=" * 50)

    df = pd.DataFrame(columns=['content', 'class'])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} not found!")

    categories = []
    total_files = 0

    for directory in os.listdir(data_path):
        directory_path = os.path.join(data_path, directory)
        if os.path.isdir(directory_path):
            categories.append(directory)
            total_files += len([f for f in os.listdir(directory_path) if f.endswith('.txt')])

    if verbose:
        print(f"Found {len(categories)} categories:")
        for cat in categories:
            print(f"   {cat}")
        print(f"Total files to process: {total_files}")

    processed_files = 0

    for directory in categories:
        directory_path = os.path.join(data_path, directory)
        if os.path.isdir(directory_path):
            category_files = 0

            for filename in os.listdir(directory_path):
                if filename.endswith('.txt'):
                    filepath = os.path.join(directory_path, filename)

                    with open(filepath, 'rb') as f:
                        content = f.read()
                        suggestion = UnicodeDammit(content)
                        encoding = suggestion.original_encoding or 'utf-8'

                    try:
                        with open(filepath, encoding=encoding) as f:
                            content = f.read()
                            current_df = pd.DataFrame({
                                'content': [content],
                                'class': [directory]
                            })
                            df = pd.concat([df, current_df], ignore_index=True)
                            category_files += 1
                            processed_files += 1
                    except Exception as e:
                        if verbose:
                            print(f"Error reading {filepath}: {e}")

            if verbose:
                print(f"   {directory}: {category_files} files loaded")

    initial_size = len(df)
    df = df.drop_duplicates(subset=['content'])
    final_size = len(df)

    if verbose:
        print(f"\nDataset Summary:")
        print(f"   Total documents: {final_size}")
        print(f"   Duplicates removed: {initial_size - final_size}")
        print(f"   Class distribution:")

        class_counts = df['class'].value_counts()
        for class_name, count in class_counts.items():
            print(f"      {class_name}: {count} documents")

    return df


def comprehensive_text_clean(df, column='content', verbose=True):
    if verbose:
        print("\nComprehensive Text Cleaning Pipeline")
        print("=" * 50)

    df_clean = df.copy()
    original_length = len(df_clean)

    if verbose:
        print("Step 1: Removing HTTP links...")
    df_clean[column] = df_clean[column].str.replace(
        r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
        '', regex=True
    )
    print(len(df_clean))

    if verbose:
        print("Step 2: Normalizing line endings...")
    df_clean[column] = df_clean[column].str.replace(r'[\r\n]+', ' ', regex=True)
    print(len(df_clean))

    if verbose:
        print("Step 3: Removing numbers...")
    df_clean[column] = df_clean[column].str.replace(r'[\w]*\d+[\w]*', '', regex=True)
    print(len(df_clean))

    if verbose:
        print("Step 4: Removing punctuation...")
    df_clean[column] = df_clean[column].str.replace(r'[^\w\s]', ' ', regex=True)
    print(len(df_clean))

    if verbose:
        print("Step 5: Normalizing whitespace...")
    df_clean[column] = df_clean[column].str.replace(r'\s+', ' ', regex=True)
    df_clean[column] = df_clean[column].str.strip()
    print(len(df_clean))

    if verbose:
        print("Step 6: Converting to lowercase...")
    df_clean[column] = df_clean[column].str.lower()
    print(len(df_clean))

    if verbose:
        print("Step 7: Removing empty documents...")
    #df_clean = df_clean[df_clean[column].str.len() > 0]

    if verbose:
        print("Step 8: Removing stopwords...")

    stop_words = STOP_WORDS

    def remove_stopwords(text):
        if pd.isna(text):
            return ""
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    df_clean[column] = df_clean[column].apply(remove_stopwords)

    print(len(df_clean))

    if verbose:
        print("Step 9: Lemmatizing words...")

    def lemmatize_text(text):
        if pd.isna(text) or text == "":
            return ""
        try:
            doc = nlp(text)
            lemmatized = [token.lemma_ for token in doc
                          if not token.is_stop and not token.is_punct and not token.is_space]
            return ' '.join(lemmatized)
        except Exception:
            return text
        
    print(len(df_clean))

    df_clean[column] = df_clean[column].apply(lemmatize_text)

    print(len(df_clean))

    #df_clean = df_clean[df_clean[column].str.len() > 10]

    final_length = len(df_clean)

    if verbose:
        print("\nText cleaning completed!")
        print(f"   Documents processed: {original_length}")
        print(f"   Documents remaining: {final_length}")
        print(f"   Documents removed: {original_length - final_length}")
        print(f"   Retention rate: {final_length/original_length*100:.1f}%")

    return df_clean


def analyze_text_statistics(df, column='content', class_column='class', verbose=True):
    if verbose:
        print("\nText Statistics Analysis")
        print("=" * 50)

    stats = {}
    df_temp = df.copy()
    df_temp['word_count'] = df_temp[column].str.split().str.len()
    df_temp['char_count'] = df_temp[column].str.len()
    
    def count_sentences(text):
        if pd.isna(text):
            return 0
        doc = nlp(text)
        return len(list(doc.sents))
    
    df_temp['sentence_count'] = df_temp[column].apply(count_sentences)

    stats['basic'] = {
        'total_documents': len(df_temp),
        'avg_words_per_doc': df_temp['word_count'].mean(),
        'avg_chars_per_doc': df_temp['char_count'].mean(),
        'avg_sentences_per_doc': df_temp['sentence_count'].mean(),
        'std_words_per_doc': df_temp['word_count'].std(),
        'min_words_per_doc': df_temp['word_count'].min(),
        'max_words_per_doc': df_temp['word_count'].max()
    }

    if verbose:
        print("Basic Statistics:")
        print(f"   Total documents: {stats['basic']['total_documents']}")
        print(f"   Average words per document: {stats['basic']['avg_words_per_doc']:.1f}")
        print(f"   Average characters per document: {stats['basic']['avg_chars_per_doc']:.1f}")
        print(f"   Average sentences per document: {stats['basic']['avg_sentences_per_doc']:.1f}")
        print(f"   Word count range: {stats['basic']['min_words_per_doc']} - {stats['basic']['max_words_per_doc']}")

    if class_column in df_temp.columns:
        class_stats = df_temp.groupby(class_column).agg({
            'word_count': ['mean', 'std', 'min', 'max'],
            'char_count': ['mean', 'std'],
            'sentence_count': ['mean', 'std']
        }).round(2)

        stats['by_class'] = class_stats

        if verbose:
            print("\nStatistics by Class:")
            print(class_stats)

    all_words = []
    for text in df_temp[column]:
        if pd.notna(text):
            all_words.extend(text.split())

    word_freq = Counter(all_words)
    stats['vocabulary'] = {
        'total_words': len(all_words),
        'unique_words': len(word_freq),
        'vocabulary_richness': len(word_freq) / len(all_words) if all_words else 0,
        'most_common_words': word_freq.most_common(10)
    }

    if verbose:
        print("\nVocabulary Analysis:")
        print(f"   Total words: {stats['vocabulary']['total_words']:,}")
        print(f"   Unique words: {stats['vocabulary']['unique_words']:,}")
        print(f"   Vocabulary richness: {stats['vocabulary']['vocabulary_richness']:.4f}")
        print("   Most common words:")
        for word, freq in stats['vocabulary']['most_common_words']:
            print(f"      {word}: {freq:,}")

    return stats


def compare_preprocessing_steps(df, column='content', steps_to_compare=None):
    if steps_to_compare is None:
        steps_to_compare = ['original', 'lowercase', 'no_punctuation', 'no_stopwords', 'lemmatized']

    print("\nPreprocessing Steps Comparison")
    print("=" * 50)

    sample_text = df[column].iloc[0]
    comparison = {}

    comparison['original'] = sample_text
    comparison['lowercase'] = sample_text.lower()
    comparison['no_punctuation'] = re.sub(r'[^\w\s]', ' ', comparison['lowercase'])
    comparison['no_punctuation'] = re.sub(r'\s+', ' ', comparison['no_punctuation']).strip()

    stop_words = STOP_WORDS
    words = comparison['no_punctuation'].split()
    filtered_words = [word for word in words if word not in stop_words]
    comparison['no_stopwords'] = ' '.join(filtered_words)

    try:
        doc = nlp(comparison['no_stopwords'])
        lemmatized = [token.lemma_ for token in doc
                      if not token.is_stop and not token.is_punct and not token.is_space]
        comparison['lemmatized'] = ' '.join(lemmatized)
        
        print("\nLemmatization Examples:")
        demo_examples = [
            "running", "better", "companies", "studies", 
            "played", "children", "feet", "mice"
        ]
        
        for word in demo_examples:
            doc_word = nlp(word)
            lemma = doc_word[0].lemma_
            print(f"   {word} → {lemma}")
            
    except BaseException:
        comparison['lemmatized'] = comparison['no_stopwords']

    for step in steps_to_compare:
        if step in comparison:
            print(f"\n{step.upper()}:")
            print(f"   Length: {len(comparison[step])} characters")
            print(f"   Words: {len(comparison[step].split()) if comparison[step] else 0}")
            print(f"   Text: {comparison[step][:200]}...")

    return comparison


def plot_preprocessing_comparison(df, column='content', class_column='class', figsize=(16, 12)):
    print("\nCreating Preprocessing Comparison Plots")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Text Preprocessing Analysis Dashboard', fontsize=16, fontweight='bold')

    original_stats = analyze_text_statistics(df, column, class_column, verbose=False)

    ax1 = axes[0, 0]
    word_counts = df[column].str.split().str.len()
    ax1.hist(word_counts, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Word Count Distribution')
    ax1.set_xlabel('Words per Document')
    ax1.set_ylabel('Frequency')
    ax1.axvline(word_counts.mean(), color='red', linestyle='--', label=f'Mean: {word_counts.mean():.1f}')
    ax1.legend()

    ax2 = axes[0, 1]
    char_counts = df[column].str.len()
    ax2.hist(char_counts, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Character Count Distribution')
    ax2.set_xlabel('Characters per Document')
    ax2.set_ylabel('Frequency')
    ax2.axvline(char_counts.mean(), color='red', linestyle='--', label=f'Mean: {char_counts.mean():.1f}')
    ax2.legend()

    ax3 = axes[0, 2]
    class_counts = df[class_column].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    wedges, texts, autotexts = ax3.pie(class_counts.values, labels=class_counts.index,
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Class Distribution')

    ax4 = axes[1, 0]
    df_temp = df.copy()
    df_temp['word_count'] = df_temp[column].str.split().str.len()
    classes = df_temp[class_column].unique()
    word_counts_by_class = [df_temp[df_temp[class_column] == cls]['word_count'].values for cls in classes]
    ax4.boxplot(word_counts_by_class, labels=classes)
    ax4.set_title('Word Count by Class')
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Word Count')
    plt.setp(ax4.get_xticklabels(), rotation=45)

    ax5 = axes[1, 1]
    all_words = []
    for text in df[column]:
        if pd.notna(text):
            all_words.extend(text.split())

    word_freq = Counter(all_words)
    top_words = word_freq.most_common(10)
    words, counts = zip(*top_words)
    ax5.barh(range(len(words)), counts, color='lightgreen')
    ax5.set_yticks(range(len(words)))
    ax5.set_yticklabels(words)
    ax5.set_title('Most Common Words')
    ax5.set_xlabel('Frequency')

    ax6 = axes[1, 2]
    vocab_richness = []
    class_names = []
    for cls in df[class_column].unique():
        class_texts = df[df[class_column] == cls][column]
        all_class_words = []
        for text in class_texts:
            if pd.notna(text):
                all_class_words.extend(text.split())
        if all_class_words:
            unique_words = len(set(all_class_words))
            total_words = len(all_class_words)
            richness = unique_words / total_words
            vocab_richness.append(richness)
            class_names.append(cls)

    ax6.bar(class_names, vocab_richness, color='gold', alpha=0.7)
    ax6.set_title('Vocabulary Richness by Class')
    ax6.set_xlabel('Class')
    ax6.set_ylabel('Richness (Unique/Total)')
    plt.setp(ax6.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    print("\nGenerating Word Cloud...")
    all_text = ' '.join(df[column].dropna())
    plt.figure(figsize=(12, 8))
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='viridis',
                          max_words=100,
                          relative_scaling=0.5).generate(all_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Most Common Terms', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    print("Preprocessing comparison plots completed!")
