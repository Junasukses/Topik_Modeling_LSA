#!/usr/bin/env python
# coding: utf-8

# # Topik Modeling Dengan Python

# topic modeling merupakan suatu pendekatan untuk menganalisis kumpulan dokumen berbentuk teks dan mengelompokkan menjadi beberapa topik. Pendekatan tersebut masuk dalam pendekatan Clustering dalam studi Machine Learning. Adapun tahap-tahapnya yaitu : 
# <ol>
#     <li>Crawling Data</li>
#     <li>Preprocessing Data</li>
#     <li>LSA</li>
# </ol>

# # Crawling Data

# Crawling data adalah suatu teknik untuk mengumpulkan data secara cepat dengan menggunakan url sebagai target data yang akan dikumpulkan. Untuk mengumpulkan data bisa menggunakan berbagai tools atau library yang ada, salah satunya adalah Scrappy. Scrapy adalah framework dari python yang berspesialis dalam melakukan web scraping dalam sekala besar, untuk menggunakan scrapy pertama kita install dahulu Scrapy dengan menggunakan pip

# In[1]:


pip install Scrapy


# Setelah menginstall Scrapy, selanjutnya import library yang dibutuhkan

# In[2]:


import scrapy
import nltk
import re


# Sesudah import library yang dibutuhkan, selanjutnya melakukan tahap scraping. Disini tahap Scrape saya simpan di class QuotesSpider dengan parameter scrapy.spider. Variabel start_urls berfungsi untuk menampung target url, dimana start_url akan mendapatkan data dari tahap looping "for page in range(1,10)". Function parse memiliki peran melakukan scrap pada element html mana, sedangkan function parse_detail memiliki peran untuk menargetkan secara spesifik seperti : 
# <ul>
#     <li>Mengambil text htmlnya atau mengambil Linknya</li>
#     <li>Membuang elemen yang tidak digunakan</li>
#     <li>Mereplace kata yang tidak digunakan dengan kata yang ingin digunakan</li>
# </ul>

# In[3]:


class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = []
    def __init__(self):
        url = 'https://pta.trunojoyo.ac.id/welcome/index/'
        for page in range(1,10):
            self.start_urls.append(url+str(page))

    def parse(self, response):
        for detail in response.css('a.gray.button::attr(href)'): 
            yield response.follow(detail.get(), callback = self.parse_detail)

    def parse_detail(self, response):
        for data in response.css('#content_journal > ul > li'):
            yield{
                'Judul': data.css('div:nth-child(2) > a::text').get(),
                'Penulis': data.css('div:nth-child(2) > span::text').get().replace('Penulis : ', ''),
                'Dospem 1': data.css('div:nth-child(3) > span::text').get().replace('Dosen Pembimbing I : ', ''),
                'Dospem 2': data.css('div:nth-child(4) > span::text').get().replace('Dosen Pembimbing II :', ''),
                'Abstraksi': data.css('div:nth-child(2) > p::text').get().replace('\n\n|\n','').replace('ABSTRAK', ''),
                'Abstraction': data.css('div:nth-child(4) > p::text').get().replace('\n\n|\n','').replace('ABSTRACT', ''),
                'Link Download': data.css('div:nth-child(5) > a:nth-child(1)::attr(href)').get().replace('.pdf-0.jpg', '.pdf'),
            }


# Silahkan save codenya dan buka cmd, pastikan terbuka di folder yang ada file scrapingnya. Kemudian jalankan perintah ini di cmd untuk memproses dan menyimpan ke csv "scrapy runspider namaFile.py -o namaFileKetikaDiSaveUlang.csv"

# # Preprocessing Data

# Preprocessing Data adalah suatu teknik untuk merubah data mentah atau raw data menajdi informasi yang bersih dan agar bisa digunakan untuk pengolahan lanjutan pada data mining. Pada pembahasan ini Preprocessing Data akan dilakukan dalam 2 tahap, yaitu :
# <ol>
#     <li>Stop Word</li>
#     <li>Cleaning Data</li>
# </ol>

# ## 1. Stop Word

# Stop Word adalah tahap untuk menghilangkan kata yang tidak memiliki arti, seperti preposisi, konjungsi, dan lain sebagainya. Contoh kata yang dihilangkan dari Stop Word adalah yang, di, ke, dan lainnya. Tanpa perlu berlama-lama mari langsung kepada tahap kodingnya, Karena disini saya menggunakan nltk untuk melakukan Stop Word maka kita perlu menginstall nltk terlebih dahulu 

# In[4]:


pip install --user -U nltk


# Dan tidak lupa untuk menginstall pandas juga untuk membantu kita dalam mengolah file

# In[5]:


pip install -U scikit-learn


# Sesudah menginstall semua yang dibutuhkan, selanjutnya kita import library yang dibutuhkan

# In[24]:


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# Sesudah mengimport yang dibutuhkan, selanjutnya kita load data yang sudah kita crawling tadi. Karena tadi hasil yang saya save dengan nama **crawlingpta.csv** maka pada saat load dengan pandas yang saya tuju adalah file **crawlingpta.csv**

# In[7]:


jurnal = pd.read_csv('crawlingpta.csv')


# Sesudah meload data selanjutnya memilih kolom yang ingin di proses, disini saya akan memproses kolom **abstraksi**, dan pada kolom itu juga saya akan menghilangkan angka yang akan mengganggu. Tahap ini juga termasuk dalam bagian Cleaning Data, tahap ini saya lakukan di awal karena kalau udah masuk ke stop word akan susah di proses. Untuk melakukannya saya buat function yang bernama **remove_number** dan di function ini akan mengembalikan nilai berupa text dimana jika ada angka akan dihapus, dan ketika memanggil kolom dikasih apply dan memanggil functionnya

# In[8]:


def remove_number(text):
    return  re.sub(r"\d+", "", text)

pre_abstrak = jurnal['Abstraksi'].apply(remove_number)
pre_abstrak


# Kemudian langkah sebelum memasuki stop word adalah harus tokenize kalimat dahulu, tokenize adalah proses untuk membagi kalimat ke dalam bagian bagian tertentu

# In[9]:


word_tokens = pre_abstrak.apply(word_tokenize)
word_tokens


# Langkah selanjutnya adalah Stop Word. Karena disini saya menggunakan nltk maka harus menentukan dahulu bahasa yang digunakan untuk menentukan bahasa menggunakan **stopwords.words('indonesian')**. Kemudian jika dirasa list stop word masih ada yang kurang maka kita bisa menambahkan sendiri dengan cara membuat list kata yang tidak ada di stop word kemudian kita extend dengan list yang kita buat sendiri **stop_words.extend(list)**

# In[15]:


def stop_w(word):
    stop_words = stopwords.words('indonesian')
    list = ['a','aajaran','aanslag','aatau','ah','abstak','abstrack','abstract','abstrak','z']
    stop_words.extend(list)
    removed = [w for w in word if w not in stop_words]
    cleaned_text=" ".join(removed)
    return cleaned_text

after = word_tokens.apply(stop_w)
after


# Untuk logika pada saat stop word sendiri sebagai berikut. Pertama membuat fungsi untuk di apply dengan variabel yang menyimpan tokens tadi. Kemudian pada fungsinya kita set bahasa stop words yang digunakan yaitu **indonesian**. Jika ada list stop words yang tidak ada pada stop words yang disediakan oleh nltk, kita bisa menambahkannya dengan cara membuat list kata yang mau dihilangkan kemudian pada stop wordsnya di extend dengan list yang menyimpan list kata yang ingin dihapus. Kemudian logika untuk perulangannya yaitu ini akan dilooping pada array dengan perulangan pada parameter fungsi dan dikasih logika percabangan jika katanya tidak ada pada list stop wordsnya maka akan masuk. Dan untuk mengembalikan agar menyatu jadi kalimat dengan cara set string kosongan kemudian join dengan variabe yang menyimpan list tadi dan terakhir di return

# ## 2. Cleaning Data

# Cleaning Data adalah proses untuk membersihkan data yang ada menjadi data yang bisa diolah. Data yang dibersihkan seperti missing value atau data kosong, karakter asing, menghilangkan angka, dan lain sebaginaya. Untuk proses penghilangan angka sudah dilakukan ketika memilih tabel **abstraksi**, maka sekarang tinggal menghilangkan karakter asing dan sekawannya. Untuk melakukan itu kita bisa menggunakan library string.punctuation. Dimana ia akan menghilangkan karakter asing yang ada

# In[19]:


def cleaning_data(data):
    removed = [d for d in data if d not in string.punctuation]
    cleaned_text=" ".join(removed)
    return cleaned_text

clearData = word_tokens.apply(cleaning_data)
clearData


# Logika dari code diatas sama seperti proses stop words dimana membuat function dahulu kemudian di apply pada variabel yang menyimpan data sebelumnya. Pada fungsinya melakukan perulangan di dalam array dimana jika kata tidak ada dalam string.punctuation maka akan masuk. Kemudian untuk di joinkan kembali dalam kalimat dengan cara string kosong kemudian di joinkan dan yang terakhr di return

# # Modeling (LSA)

# LSA merupakan metode yang memanfaatkan model statistik matematis untuk menganalisa struktur semantik suatu teks. LSA bisa digunakan untuk menilai esai dengan mengkonversikan esai menjadi matriks-matriks yang diberi nilai pada masing-masing term untuk dicari kesamaan dengan term referensi. Sebelum masuk ke materi LSA mari kita cari tf idfnya, untuk mencari tf-idf kita bisa menggunakan bantuan dari sklearn. Berikut cara menginstall sklearn

# In[38]:


pip install -U scikit-learn


# Sesudah install selanjutnya import **TfidfVectorizer** yang berada di **sklearn.feature_extraction.text**

# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer


# untuk menggunakan **TfidfVectorizer** tinggal memanggilnya saja kemudian untuk parameternya bisa kita set mau max featuresnya berapa. Sesudah memanggil **TfidfVectorizer** selanjutnya kita fit_transform dengan parameter data yang sudah di preprocessing tadi

# In[40]:


vect = TfidfVectorizer(max_features=1000)
vect_text = vect.fit_transform(clearData)
print(vect_text.shape)
print(vect_text)


# Sesudah mencari tf selanjutnya mencari idfnya, untuk mencari ifg tinggal memanggil **vect.idf_** dan dimasukan kedalam **dict(zip())**

# In[32]:


idf=vect.idf_
dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['dan'])
print(dd['cumi'])


# Berdasarkan data diatas, __dan__ merupakan kata yang paling sering muncul dan __cumi__ merupakan kata yang jarang muncul. Sesudah mencari tf-idf, selanjutnya kita masuk ke tahap LSAnya. Untuk LSA disiini saya menggunakan bantuan dari modulnya sklearn yaitu **TruncatedSVD**. Pertama tama di import dahulu modulnya dari **sklearn.decomposition**

# In[41]:


from sklearn.decomposition import TruncatedSVD


# Selanjutnya kita cari topik yang sering di temui 

# In[43]:


lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)
lsa_top=lsa_model.fit_transform(vect_text)

print(lsa_top)
print(lsa_top.shape)


# In[35]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# In[36]:


print(lsa_model.components_.shape)
print(lsa_model.components_)


# Setelah melalui proses yang panjang didapatkan sebuah list dari kata-kata yang penting dan memiliki makna dari setiap 10 topic yang ditampilkan. Sederhananya dibawah ini ditampilkan 10 kata dalam setiap topic.

# In[37]:


vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

