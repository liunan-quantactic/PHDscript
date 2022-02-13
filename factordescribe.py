'''This moudel is to test the factor's effects'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''temporary demo
   from quantactic import sql_connector,initialdata
   import sys
   sys.path.append("G:\\moudles")
   user='root'
   pw='7026155@Liu'
   h='127.0.0.1'
   p=3306
   sch = 'astocks'
   engine = sql_connector(user,pw,h,p,sch)
   query ='select td,codenum,chg,close,total_share from astocks.market where td>\''+str(20190101)+'\';'
   df=pd.read_sql(query,engine,index_col='td',parse_dates=True)
   facname='chg'
   n=3
   '''


def factordescribe(df,facname,n,weight='equal'):
    '''the argument of df should include factors and  change of price and weight vector if required'''
    '''df--the original data
       facname--factor name type:str
       n--the group number
       weight--the weight of component of portfolio'''
    '''step 2 按照因子大小分位数分组'''
    #得到所需要的时间
    period = np.unique(df.index)
    all_return = []
    for t in period:
        tdf = df.loc[t]
        classes = np.array(pd.qcut(tdf[facname],n,['P1','P2','P3']).tolist())
        tdf.insert(0,'class',classes)
        gdf = tdf.groupby('class')
        T_port_return=[]
        for key in ['P1','P2','P3']:#不能用df.group.keys,顺序会错
            port = gdf.get_group(key)
            cap = port['close']*port['total_share']
            all_cap = cap.sum()
            weight = cap/all_cap
            port_return=np.sum(port['chg']*weight)
            T_port_return.append(port_return)
        all_return.append(T_port_return)
    all_return = pd.DataFrame(all_return,index=pd.to_datetime(period.astype('str'),format='%Y%m%d'),columns=['P1','P2','P3'])
    all_return.insert(3,'P3-P1',all_return['P3']-all_return['P1'])
    rall_return = all_return.rolling(5).apply(lambda x:np.cumprod(1+x/100))
    fig=all_return[['P1','P2','P3']].plot(title='performance',style=['r:.','g-.x','b--^'])
