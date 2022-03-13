from selenium import webdriver
import requests
import pandas as pd
import time,os
import random

def DownloadFile(mp3_url,save_path="",mp3_name=''):
    # 读取MP3资源
    try:
        res = requests.get(mp3_url,stream=True)
        time.sleep(random.randint(3, 8))
        # 获取文件地址
        file_path = os.path.join(save_path, mp3_name+'.mp3')
        print('开始写入文件：', file_path)
        # 打开本地文件夹路径file_path，以二进制流方式写入，保存到本地
        with open(file_path, 'wb') as fd:
            for chunk in res.iter_content():
                fd.write(chunk)
        print(mp3_name+' 成功下载！')
    except BaseException as e:
        print(e)
        print("程序错误")

def get_tatoeba_urls(driver,num=18):
    #os.makedirs(save_path,exist_ok=True)
    out = []
    for i in range(0,num):
        u='https://tatoeba.org/zh-cn/audio/index/cmn?page=%s'%(i+1)
        print(u)
        driver.get(u)
        time.sleep(random.randint(6,15))
        a1=driver.find_element_by_id("main_content")
        #a2=a1.find_elements_by_xpath('//section//div[@class="sentence mainSentence"]')
        a4 = a1.find_elements_by_xpath('//section//div[@class="sentence mainSentence"]//div[@class="sentenceContent"]')
        a5 = a1.find_elements_by_xpath('//section//div[@class="sentence mainSentence"]//div[@class="audio column"]//a')
        X1=[(x.get_attribute('data-sentence-id'),x.text) for x in a4]
        X2 = [(x.get_attribute('title'), x.get_attribute('href')) for x in a5]
        print(len(X1),len(X2))
        for x1,x2 in zip(X1,X2):
            sentence_id=x1[0]
            text = x1[1]
            audio_url=x2[1]
            user = x2[0].replace('播放音频','').replace('的录音','')
            #mp3_name=user+'_'+sentence_id
            #print(sentence_id,user,text,audio_url)
            out.append((sentence_id, user, text, audio_url))
            #if i>11:
            #    DownloadFile(mp3_url=audio_url, save_path=save_path, mp3_name=mp3_name)

    out=pd.DataFrame(out,columns=['sentence_id','user','text','audio_url'])
    out=out.drop_duplicates()
    print(out.shape)
    return out

def download_mp3(X,save_path='tatoeba_cn'):
    os.makedirs(save_path, exist_ok=True)
    n=X.shape[0]
    for i in range(0,n):
        sentence_id=X['sentence_id'].iloc[i]
        user = X['user'].iloc[i]
        audio_url=X['audio_url'].iloc[i]
        mp3_name = user + '_' + str(sentence_id)
        DownloadFile(mp3_url=audio_url, save_path=save_path, mp3_name=mp3_name)



if __name__ == '__main__':
    driver = webdriver.Chrome()
    X=get_tatoeba_urls(driver, num=18)
    driver.close()
    driver.quit()
    X.to_csv('tatoeba_cn_%s.csv' %X.shape[0],index=False)
    download_mp3(X, save_path='tatoeba_cn')
