import cvxopt
import numpy as np
from cvxopt import matrix,solvers
import pandas as pd
import statsmodels.api as sm

#全部数据
df = pd.read_csv("F:\\all_factor_month.csv",index_col=0,parse_dates=True)
#获得目标数据
need_df=df[["chg","codenum","beta","EBITDA2EV","free_EV","MOMOM","STOM","risk_aver"]]
need_df=need_df[need_df["codenum"].isna()==False]
#-------------------求因子之间的相关性---------------------------------
#按时间排列因子
period=np.unique(need_df.index)
mean=[]
for t in period:
    mean.append(need_df[["beta","EBITDA2EV","free_EV","MOMOM","STOM","risk_aver"]].loc[t].mean(axis=0))
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
    R1m=-1*np.diag(np.ones(len(codes)))
    R2m=np.diag(np.ones(len(codes)))
    G=matrix(np.vstack([R1m,R2m]),tc='d')
    h=matrix(np.hstack([np.zeros(len(codes)),np.array([0.05]*len(codes))]),tc='d')
    A=matrix(np.ones(len(codes)),tc='d').T
    b=matrix(1,tc='d')
    sol = solvers.qp(XXBQB, c, G, h, A, b)
    w.append(list(sol['x']))

W_all_NRA=pd.DataFrame(w,index=p_chg_df.index,columns=codes)

#加权计算收益
W_all_NRA=pd.read_csv("F:\\weight_all.csv",index_col=0,parse_dates=True)
codes=W_all_NRA.columns
df = pd.read_csv("F:\\all_factor_month.csv",index_col=0,parse_dates=True)
req_df=df[df['codenum'].isin(codes)]
chg_df=pd.pivot_table(req_df,index=req_df.index,columns='codenum',values='chg')
chg_df=chg_df.fillna(0)
chg_df[chg_df>2]=0
port_return=[]
for t in W_all_NRA.index:
    tchg=chg_df.loc[t]
    tweight=W_all_NRA.loc[t]
    w_return=tchg*tweight
    port_return.append(np.sum(w_return))
#市值加权
w_df=pd.pivot_table(req_df,index=req_df.index,columns='codenum',values='total_EV')
w_df=w_df.fillna(0)
all_EV=w_df.sum(axis=1)
port_return=[]
for t in W_all_NRA.index:
    tchg=chg_df.loc[t]
    t_EV=w_df.loc[t]
    tweight=t_EV/all_EV.loc[t]
    w_return=tchg*tweight
    port_return.append(np.sum(w_return))


port_return_df=pd.DataFrame(port_return,index=W_all_NRA.index)
port_return_df.to_csv("F:\\port_return_EV.csv")

##################################################
port_return_df=pd.read_csv("G:\\port_return.csv",index_col=0,parse_dates=True)
port_cum_df=np.cumprod(1+port_return_df)
port_cum_df.plot(color=['g','b','r'],style=['-.','--','-'])