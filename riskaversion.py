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


#宏观经济数据
excel_df=pd.read_excel("G:\\BHSdata.xlsx",index_col=0)
excel_df.index = excel_df.index.strftime("%Y-%m")
#市场数据，不包括科创板
m_query="select td,codenum,chg from market where td>20060630 and td<20210730 and (codenum like '00%' or codenum like '30%'" \
        " or codenum like '60%');"
mdf=pd.read_sql(m_query,engine,index_col='td')

#将涨幅过高的数据删掉（就是有些新股首日以及数据长度太短的删掉）
#透视表
pmdf=pd.pivot_table(mdf,index=mdf.index,columns="codenum",values='chg')
pmdf=pmdf/100  #单位修正
pmdf.fillna(0,inplace=True)
#计算累计净值
netvalue=np.cumprod(1+pmdf,axis=0)
netvalue_MA5=netvalue.rolling(5).mean()
netvalue_MA10=netvalue.rolling(10).mean()
netvalue_MA20=netvalue.rolling(20).mean()
#开始个股处理
all_code = netvalue.columns
for c in all_code:
    cdf = pd.concat([netvalue[c],netvalue_MA5[c],netvalue_MA10[c],netvalue_MA10[c]],axis=1)
    cdf.fillna(0,inplace=True)
    cdf_std = cdf.std(axis=1)
    bench_Z = netvalue[c][cdf_std<0.01]
    bench_Z = bench_Z.reindex(netvalue[c].index,method='ffill')#数据对齐
    prior_z = bench_Z/netvalue[c]
    prior_z.fillna(1,inplace=True)#将没有数据的地方填充成1
    df_need=pd.concat([netvalue[c],bench_Z,prior_z],axis=1)
    #抽取月底数据
    df_need.index=pd.to_datetime(df_need.index,format="%Y%m%d")
    month=np.unique(df_need.index.strftime("%Y-%m"))#获得所有的月份
    #逐个月份提取数据
    monthend=[]
    for m in month:
        monthend.append(df_need[m].index[-1].strftime("%Y-%m-%d"))
    #得到每个月的月底数据
    df_month_need = df_need.reindex(pd.to_datetime(monthend, format="%Y-%m-%d"))
    df_month_need.columns=['Price','Z','ZtoPrice']
    #加入宏观数据
    df_month_need.index = df_month_need.index.strftime("%Y-%m")
    df_month_need = df_month_need.reindex(excel_df.index)
    df_month_need[['CPI','rf']] = excel_df
    df_month_need.dropna(axis=0,inplace=True)
    #需要错开时间处理（非常重要）
    df_new_month=df_month_need.iloc[:-1]
    df_new_month.index=df_month_need.index[1:]
    df_new_month.insert(3,'R_t1',df_month_need.Price.pct_change())
    consumption=np.cumprod(1+df_new_month.CPI)#consumption-消费
    rt = df_new_month.R_t1#区间收益（就是这个月的收益）
    #投资效用
    uv = (1+df_new_month.rf)-rt*(np.power(consumption,0.6)/0.6)
    df_new_month.insert(6,"utility",uv)
    risk_aver=[]
    for td in df_new_month.index:
        if df_new_month.ZtoPrice.loc[td] ==1:
            if df_new_month.R_t1.loc[td]>=0:
                risk_aver.append(1)
            else:
                risk_aver.append(1+df_new_month.utility.loc[td]/np.power(-df_new_month.Price.loc[td]
                                                                       *df_new_month.R_t1.loc[td],0.5))
        elif df_new_month.ZtoPrice.loc[td] <=1:
            if df_new_month.R_t1.loc[td]>=df_new_month.rf.loc[td]:
                risk_aver.append(1)
            else:
                temp_uv1=df_new_month.utility.loc[td]-np.power(df_new_month.Price.loc[td]-df_new_month.Z.loc[td],0.5)
                temp_uv2=abs(df_new_month.Z.loc[td]*df_new_month.rf.loc[td]-(df_new_month.Price.loc[td]*df_new_month.R_t1.loc[td]))
                risk_aver.append(1+temp_uv1/np.power(temp_uv2,0.5))
        else:
            if df_new_month.R_t1.loc[td]>=0:
                risk_aver.append(1)
            elif df_new_month.R_t1.loc[td]<0:
                part_risk_av=df_new_month.utility.loc[td] / np.power(-df_new_month.Price.loc[td]* df_new_month.R_t1.loc[td], 0.5)
                risk_aver.append(1+part_risk_av+3*(df_new_month.ZtoPrice.loc[td]-1))
    df_new_month.insert(7, "risk_av", risk_aver)
    #df_new_month.to_csv("F:\\111.csv")
    print(np.corrcoef(df_new_month.R_t1, df_new_month.risk_av)[0][1])



