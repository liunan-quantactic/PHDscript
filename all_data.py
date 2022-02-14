#该模块是为了提取数据
import numpy as np
import pandas as pd
import os
import datetime

#定义log函数
def mylog(logpath='F:\\'):
    '''this moduel is to avoid printing log repeatly'''
    import logging.handlers
    logger = logging.getLogger('DBA')#logger名称
    if 'log' not in os.listdir(logpath):#设置log的存储路径
        final_path=os.mkdir(logpath+'log')
    else:
        final_path =logpath+'log'
    final_log=final_path+'\\log_'+datetime.datetime.now().strftime('%Y%m%d-%H')+'.log'#设定log的文件名,该log为完整的log
    error_log=final_path+'\\error_log_'+datetime.datetime.now().strftime('%Y%m%d-%H')+'.log'#设定error_log,该log只有出现错误才写入
    if not logger.handlers: #如果logger.handlers列表为空，则添加，否则，直接去写日志，否则会出现重复记录
        logger.setLevel(logging.DEBUG)#设定DEBUG以上的等级的log才会处理
        handler1 = logging.FileHandler(final_log)#log输出到文件的handler
        handler1.setLevel(logging.DEBUG)
        handler2 = logging.FileHandler(error_log)#出现错误的log输出到文件的handler
        handler2.setLevel(logging.ERROR)#log级别为ERROR
        handler3 = logging.StreamHandler()#log输出到屏幕的handler
        handler3.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')#log的格式
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        handler3.setFormatter(formatter)
        logger.addHandler(handler1)#添加handler
        logger.addHandler(handler2)
        logger.addHandler(handler3)
    return logger
#定义sql的接口函数
def sql_connector(username,password,server,port,schema):
    #########################################################
    #使用Mysql.connector 链接Mysql server
    #
    #
    #########################################################
    from sqlalchemy import create_engine
    '''This tool is for creating a connection with Mysql server'''
    #write connector string as fromat mysql+engine://username:password@adress:port/schema
    con_str = 'mysql+mysqlconnector://'+str(username)+':'+str(password)+'@'+str(server)+':'+str(port)+'/'+str(schema)
    try:
        con = create_engine(con_str)
        #check if the connection is vaild
        query = 'use '+str(schema)+';'
        con.execute(query)
        log_str='Connect to SERVER ' + str(server)+' SCHEMA '+str(schema) + ' successful !'
        mylog().info(log_str)
        return con
    except Exception as e:
        mylog().error(e)

user2='root'
pw2='7026155@Liu'
h2='127.0.0.1'
p=3306
sch2='astocks'
engine=sql_connector(user2,pw2,h2,p,sch2)


#获取目标股票名称
all_code_query="select distinct codenum from market where td>20060630 and td<20210730 and (codenum like '00%' or codenum like '30%'" \
        " or codenum like '60%');"
all_code=pd.read_sql(all_code_query,engine,index_col='codenum')
Mdf = pd.DataFrame()
for codenum in all_code:
    i = i+1 #count
    finance_query="select "