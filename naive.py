# -*- coding: utf-8 -*-
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
from collections import Counter
import re
import numpy as np
np.set_printoptions(threshold=np.inf)

def main():
    print("tunggu..")
    arraydatalatih = get_arraydatalatih_setelahprepros()
    arraydatauji= get_arraydatauji_setelahprepros()
    
    #arraydatauji   = get_arraydatauji_setelahprepros()
 
    print("\nData Latih")
    print(np.array(arraydatalatih).reshape(24,1))
    print("\nData Uji")
    print(np.array(arraydatauji).reshape(6,1))
            
    term = ((re.findall('\w+', " ".join(arraydatalatih))))
    
    #mencari kata unik
    
    unique_term = set(term) 
    print(unique_term)
    
    
    #mencari seluruh kata di setiap kategori
    array_kata_kategori0 = (" ".join(arraydatalatih[0:8]).split())
    array_kata_kategori1 = (" ".join(arraydatalatih[8:16]).split())
    array_kata_kategori2 = (" ".join(arraydatalatih[16:24]).split())
       
    #weighting
    counter2 = Counter(array_kata_kategori2)    
    counter1 = Counter(array_kata_kategori1) 
    counter0 = Counter(array_kata_kategori0) 
    
    array_term_dan_conditional_probability=[];    
    for kata in unique_term:
        peluanglabel2 = (counter2[kata]+1)/(len(array_kata_kategori2)+len(unique_term))
        peluanglabel1 = (counter1[kata]+1)/(len(array_kata_kategori1)+len(unique_term))    
        peluanglabel0 = (counter0[kata]+1)/(len(array_kata_kategori0)+len(unique_term))   
        object_word = Term(kata,counter2[kata],counter1[kata],counter0[kata],peluanglabel2,peluanglabel1,peluanglabel0)
        array_term_dan_conditional_probability.append(object_word)

    
    arraydatalatih_default = get_arraydatalatih()
    arraydatauji_default   = get_arraydatauji()
    array_term_dan_rawweighting = array_term_dan_conditional_probability
    
    f2 = len([i for i in arraydatalatih_default if (int(i.strip()[-1]))==2]);
    f1 = len([i for i in arraydatalatih_default if (int(i.strip()[-1]))==1]);
    f0 = len([i for i in arraydatalatih_default if (int(i.strip()[-1]))==0]);    

    
    p_prior_2 = f2/len(arraydatalatih_default)
    p_prior_1 = f1/len(arraydatalatih_default)
    p_prior_0 = f0/len(arraydatalatih_default)           
    
    counter = 1;
    banyak_labelbenar = 0
    banyak_labelsalah = 0
    banyak_seluruhdata = 0
    i=0;
    
    for du in arraydatauji:
        datauji = kalimat(du)
        datauji.arrayterm = set(datauji.sentence.split())
        
        sigma_conditionalprobability_setiapterm_2 = 1
        sigma_conditionalprobability_setiapterm_1 = 1
        sigma_conditionalprobability_setiapterm_0 = 1
        for x in array_term_dan_rawweighting:
            for term in datauji.arrayterm:
                if ((Term(term,None,None,None,None,None,None)).word == x.word):
                    sigma_conditionalprobability_setiapterm_2*=x.peluanglabel2
                    sigma_conditionalprobability_setiapterm_1*=x.peluanglabel1
                    sigma_conditionalprobability_setiapterm_0*=x.peluanglabel0
                else:             
                    sigma_conditionalprobability_setiapterm_2 *= 1;
                    sigma_conditionalprobability_setiapterm_1 *= 1;
                    sigma_conditionalprobability_setiapterm_0 *= 1;
        
        p_posterior_2 = p_prior_2 * sigma_conditionalprobability_setiapterm_2
        p_posterior_1 = p_prior_1 * sigma_conditionalprobability_setiapterm_1
        p_posterior_0 = p_prior_0 * sigma_conditionalprobability_setiapterm_0
        
        
        
        max_probability = max([p_posterior_2,p_posterior_1,p_posterior_0])
        if max_probability==p_posterior_2:
            datauji.label=2
        elif max_probability==p_posterior_1:
            datauji.label=1
        elif max_probability==p_posterior_0:
            datauji.label=0            
        
        if(datauji.label == int(arraydatauji_default[i].strip()[-1])):
            print("data uji ke : "+str(counter)+" memiliki kategori = "+str(datauji.label))
            banyak_seluruhdata+=1
            banyak_labelbenar+=1
        else:
            print("data uji ke : "+str(counter)+" memiliki kategori = "+str(datauji.label)+" [SALAH]")
            banyak_seluruhdata+=1
            banyak_labelsalah+=1
        print(str(p_posterior_0)+" & "+str(p_posterior_1)+" & "+str(p_posterior_2))
        i+=1        
        counter+=1
        
    print("benar   = "+str(banyak_labelbenar)) 
    print("salah   = "+str(banyak_labelsalah))
    print("akurasi = "+str(float("{0:.4f}".format(banyak_labelbenar/banyak_seluruhdata))*100)+str(" %"))
   
class kalimat():
     def __init__(self, sentence):
         self.sentence = sentence;
         self.hasilpreprocessing = None;
         self.arrayterm = None;
         self.label = None;
class Term():
    def __init__(self, word, frequency_in_c2,frequency_in_c1,frequency_in_c0,peluanglabel2,peluanglabel1,peluanglabel0):
        self.word = word;
        self.frequency_in_c2 = frequency_in_c2;
        self.frequency_in_c1 = frequency_in_c1;
        self.frequency_in_c0 = frequency_in_c0;
        self.peluanglabel2 = peluanglabel2        
        self.peluanglabel1 = peluanglabel1
        self.peluanglabel0 = peluanglabel0

def TextPreprocessing(sms):
    s2 = (re.sub("\d+", '', sms)).lower()         
    s3 = re.sub(r'\W', ' ', s2)                
    s4 = re.sub(r'\^[a-zA-Z]\s+', ' ', s3)     
    s5 = re.sub(r"\b[a-z]\b", '', s4)          
    s6 = re.sub(r'\s+', ' ', s5, flags=re.I)   
    stringStopword = open("stopword.txt", "r").read().replace('\n', ' ')
    arrayStopword = stringStopword.split(" ");
    arrayKalimat = s6.split(" ");

    x = [i for i in arrayKalimat if i not in arrayStopword]
                
    hasilStopword = " ".join(x)
    hasilStemming = stemmer.stem(hasilStopword)
    return hasilStemming

def get_arraydatalatih():
    stringdataLatih = open("datalatih.txt", "r",encoding='utf-8').read()
    arraydataLatih = stringdataLatih.split("\n");
    return arraydataLatih  
  
def get_arraydatalatih_setelahprepros():
    stringdataLatih = open("datalatih.txt", "r",encoding='utf-8').read()
    arraydataLatih = stringdataLatih.split("\n");
    for i in range (0,24):
        arraydataLatih[i] = TextPreprocessing(arraydataLatih[i])  
        
    return arraydataLatih

def get_arraydatauji():
    stringdatauji= open("datauji.txt", "r",encoding='utf-8').read()
    arraydatauji = stringdatauji.split("\n");
    return arraydatauji

def get_arraydatauji_setelahprepros():
    stringdatauji= open("datauji.txt", "r",encoding='utf-8').read()
    arraydatauji = stringdatauji.split("\n");
    for i in range (0,6):
        arraydatauji[i] = TextPreprocessing(arraydatauji[i])      
    return arraydatauji

if __name__ == "__main__":
    main()