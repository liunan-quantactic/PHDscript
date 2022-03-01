import cvxopt
import numpy as np
from cvxopt import matrix,solvers
import pandas as pd
import statsmodels.api as sm

#全部数据
df = pd.read_csv("G:\\all_factor_month.csv",index_col=0,parse_dates=True)
#获得目标数据
need_df=df[["chg","codenum","beta","EBITDA2EV","free_EV","MOMOM","STOM"]]
need_df=need_df[need_df["codenum"].isna()==False]
#-------------------求因子之间的相关性---------------------------------
#按时间排列因子
period=np.unique(need_df.index)
mean=[]
for t in period:
    mean.append(need_df[["beta","EBITDA2EV","free_EV","MOMOM","STOM"]].loc[t].mean(axis=0))
ts_df=pd.DataFrame(mean,index=period)
ts_df=ts_df.fillna(method='bfill')
Q=ts_df.corr().values
Q=matrix(Q,tc='d')
#---------------个股对每一个因子的回归系数----------------------
betas=[]
codes=[]
all_code=np.unique(need_df["codenum"])
for c in all_code:
    print(c)
    temp_df=need_df[need_df['codenum']==c]
    temp_df=temp_df.fillna(method='bfill')
    temp_df = temp_df.fillna(method='ffill')
    temp_df = temp_df.dropna()
    if len(temp_df)>=10:
        codes.append(c)
        y=temp_df['chg']
        x=temp_df[["beta","EBITDA2EV","free_EV","MOMOM","STOM"]]
        X=sm.add_constant(x,has_constant='add')
        mod=sm.OLS(y,X).fit()
        betas.append(mod.params.values)
B=pd.DataFrame(betas,index=codes)
B=B.fillna(method='ffill')


XX=pd.DataFrame(betas,index=codes,columns=["const","beta","EBITDA2EV","free_EV","MOMOM","STOM"])
XXB=XX[["beta","EBITDA2EV","free_EV","MOMOM","STOM"]].values
XXBQ=np.dot(XXB,Q)
XXBQB=np.dot(XXBQ,XXB.T)
XXBQB=matrix(XXBQB,tc="d")

p_chg_df=pd.pivot_table(need_df,index=need_df.index,columns='codenum',values='chg')
p_chg_df=p_chg_df[codes].fillna(0)

#要计算所有时刻的权重

#不带risk_aver的
#c为这些股票的收益率
w=[]
for t in p_chg_df.index:
    c=matrix(-1*p_chg_df.loc[t],tc='d')
    G=matrix(-1*np.diag(np.ones(len(codes))),tc='d')
    h=matrix(np.zeros(len(codes)),tc='d')
    A=matrix(np.ones(len(codes)),tc='d').T
    b=matrix(1,tc='d')
    sol = solvers.qp(XXBQB, c, G, h, A, b)
    w.append(list(sol['x']))

W_all_NRA=pd.DataFrame(w,index=p_chg_df.index,columns=codes)