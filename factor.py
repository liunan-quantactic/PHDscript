# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import threading
class MyThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
from quantactic import sql_connector,initialdata,mylog
import sys
sys.path.append("G:\\moudles")
user = 'root'
pw = '7026155@Liu'
h = '127.0.0.1'
p = 3306
sch = 'astocks'
engine = sql_connector(user,pw,h,p,sch)
save_path = "G:\\factors"
#get all stocks codes
code_query = 'select distinct codenum from astocks.market order by codenum;'
all_code = np.array(engine.execute(code_query).fetchall())
#得到行业指标的字典
comp_query = 'select codenum,SW_c1,SW_name_CL1 from astocks.company;'
ind_df = pd.read_sql(comp_query,engine)
ind_np = ind_df.values[:,0:2]#找到对应的代码和具体行业的对应关系
ind_class = np.unique(ind_df.values[:,1])
comp_industry_dic = dict(ind_np)
#提前得到benchmark以求得beta以及residual
index_query = 'select td,chg from astocks.indexprice where codenum=\''+'000985.CSI'+'\';'
index_df = pd.read_sql(index_query,con=engine,index_col='td')
index_df.index=pd.to_datetime(index_df.index,format='%Y%m%d')
def factormaker(codenum):
    '''get data--Factors based on data of change,close,volume,amount,total share,float share,free share,'''
    data_query = 'select td,codenum,chg,close,vol,amt,total_share,float_share,free_share ' \
                 'from market where codenum=\''+codenum[0]+'\';'
    market_df = pd.read_sql(data_query,engine,index_col='td',parse_dates=True)
    market_df.index = pd.to_datetime(market_df.index,format='%Y%m%d')
    market_df.drop_duplicates(inplace=True)
    '''SIZE factor:the Size factor equals close times different share'''
    #因子：市值因子
    total_EV = market_df['close']*market_df['total_share']*10000
    float_EV = market_df['close'] * market_df['float_share'] * 10000
    free_EV = market_df['close'] * market_df['free_share'] * 10000
    market_df['total_EV'] = total_EV
    market_df['float_EV'] = float_EV
    market_df['free_EV'] = free_EV
    SIZE = pd.concat([total_EV,float_EV,free_EV],axis=1)
    SIZE.columns=['total_EV','float_EV','free_EV']
    '''get data--Factors based on data of financial data'''
    data_query = 'select fd,codenum,ad,total_current_assets,total_noncurrent_assets,total_assets,' \
                 'total_current_liabilities,total_noncurrent_liabilities,total_liabilities,' \
                 'total_shareholders_equity_including_MI,operating_revenue,operating_profit,' \
                 'total_profit,net_profit_including_minority_interest_income,deductedprofit,' \
                 'net_operating_cashflow from finance where codenum=\''+codenum[0]+'\';'
    raw_finance_df = pd.read_sql(data_query, engine, index_col='fd', parse_dates=True)
    '''get data--Factor based on data of dividend'''
    data_query = 'select ex_dt,codenum,cash from astocks.dividend where codenum=\''+codenum[0]+'\';'
    raw_dividend_df = pd.read_sql(data_query, engine, index_col='ex_dt', parse_dates=True).fillna(0)
    raw_dividend_df.index = pd.to_datetime(raw_dividend_df.index, format='%Y%m%d')
    raw_dividend_df.drop_duplicates()
    #数据调整成两个月之后月底的数据
    finance_df = raw_finance_df.copy()
    finance_df.index = pd.to_datetime(raw_finance_df.index,format='%Y%m%d')+pd.tseries.offsets.MonthEnd(2)
    raw_finance_df.index = pd.to_datetime(raw_finance_df.index,format='%Y%m%d')
    # 开始整理因子
    #因子：
    #beta 以及 residual
    #将数据取交集
    window = np.sort(np.intersect1d(market_df.index,index_df.index))
    endogen = market_df['chg'].reindex(window)/100
    exogen = index_df['chg'].reindex(window)/100
    exogen = sm.add_constant(exogen)
    def rolling_reg(endogen,exogen,freg='M',method=None):
        #将时间调整成frequency需要的时间，目前实现月的功能
        adjust_index = endogen.resample(freg).sum().index.strftime('%Y-%m')
        params = []
        sigmas = []
        tvalues = []
        interval = []
        resid_mean = []
        resid_std = []
        for i in range(len(adjust_index)):
            if i-11 > 0:
                rolling_endogen = endogen[adjust_index[i-11]:adjust_index[i]]
                rolling_exogen = exogen[adjust_index[i-11]:adjust_index[i]]
                model = sm.OLS(rolling_endogen,rolling_exogen).fit()
                beta = model.params.values        #12个月的coefficient
                sigma = model.bse                 #12个月系数的标准差，可以选择robust
                tvalue = model.tvalues            #12个月系数的t
                residual_mean = np.mean(model.resid)    #12个月系数对应的残差均值
                residual_std = np.std(model.resid)   # 12个月系数对应的残差均值
                params.append(beta)
                sigmas.append(sigma)
                tvalues.append(tvalue)
                resid_mean.append(residual_mean)
                resid_std.append(residual_std)
                interval.append(adjust_index[i])
        beta_df = pd.DataFrame(np.column_stack([params,sigmas,tvalues,resid_mean,resid_std]),index=pd.to_datetime(interval,format='%Y%m%')+pd.tseries.offsets.MonthBegin(1),
                             columns=['incpt','beta','incpt_std','beta_std','t_incpt','t_beta','res_avrg','res_std'])
        #注：reindex的时候开始阶段有NA值
        return beta_df
    betas = rolling_reg(endogen, exogen, freg='M')
    factor_betas = betas.reindex(market_df.index,method='ffill')
    #动量 momentum
    accum_return = np.cumprod(1+market_df['chg']/100)#累计收益
    accum_return = accum_return.to_frame()
    accum_return_OM = accum_return.resample('M').last()#截至每个月底的累计收益
    MOMOM = accum_return_OM.pct_change(1)#本月的涨幅
    MOM12T2 = accum_return_OM.pct_change(12)-accum_return_OM.pct_change(1)#近12个月涨幅剔除近1个月
    MOM3T2 = accum_return_OM.pct_change(3) - accum_return_OM.pct_change(1)#近3个月涨幅剔除近1个月
    MOM = pd.concat([MOMOM,MOM3T2,MOM12T2],axis=1)
    MOM.columns = ['MOMOM','MOM3T2','MOM12T2']
    MOM=MOM.reindex(market_df.index,method='ffill')
    #book-to-price 最近一期报告的净权益除以最新的市值，记为‘BTM’
    equity = finance_df['total_shareholders_equity_including_MI'].reindex(market_df.index,method='ffill')
    BTM = equity/total_EV
    BTM = BTM.to_frame()
    BTM.columns = ['BTM']
    #流动性 最近21天个交易日的总换手率 share turnover，one month
    turnoverperday = market_df['vol']/market_df['total_share']
    STOM = turnoverperday.rolling(21).sum()
    #        最近3个月的平均换手率
    STOQ=turnoverperday.rolling(63).sum()/3
    #        最近一年的平均换手率
    STOY = turnoverperday.rolling(252).sum() / 12
    TO = pd.concat([STOM,STOQ,STOY],axis=1)
    TO.columns = ['STOM','STOQ','STOY']
    #收益指标
    #最近一个季度的经营现金流比市值
    opcf = finance_df['net_operating_cashflow'].reindex(market_df.index,method='ffill')
    CETOP = opcf/total_EV
    #EBITDA/EV
    EBITDA=finance_df['total_profit'].reindex(market_df.index,method='ffill')
    EBITDA2EV = EBITDA/total_EV
    #增长指标
    #年利润增长率 profit growth of year
    #最近四个季度的年利润TTM
    profit_TTM = raw_finance_df['net_profit_including_minority_interest_income'].rolling(4).sum()
    profit_TTM.index = pd.to_datetime(raw_finance_df.index,format='%Y%m%d')+pd.tseries.offsets.MonthEnd(2)#时间向后调整2个月
    #TTM的市盈率
    PETTM = profit_TTM.reindex(market_df.index,method='ffill')/total_EV
    #最近四个季度的利润增长GPOY
    GPOY=profit_TTM.pct_change().reindex(market_df.index,method='ffill')
    #最近三年的利润增长
    profit_3Y = raw_finance_df['net_profit_including_minority_interest_income'].resample('Y').sum()#每年的总利润
    profit_3Y = profit_3Y.to_frame()
    profit_3Y.index = profit_3Y.index+pd.tseries.offsets.MonthEnd(2)#再调整成为报告期
    GPO3Y = profit_3Y.pct_change(3).reindex(market_df.index,method='ffill')
    #现金分红比价格
    CD2EV = raw_dividend_df['cash'].reindex(market_df.index,method='ffill')/market_df['close']
    PORFIT = pd.concat([CETOP,EBITDA2EV,PETTM,GPOY,GPO3Y,CD2EV],axis=1)
    PORFIT.columns=['CETOP','EBITDA2EV','PETTM','GPOY','GPO3Y','CD2EV']
    #纯财务指标
    #杠杆指标（长期负债）
    TD = finance_df['total_liabilities']
    #负债市值比
    MLEV = TD.reindex(market_df.index,method='ffill')/total_EV
    #资产负债率及变化
    DTOA = TD/finance_df['total_assets']
    DTOADIF = DTOA.pct_change()
    DTOA = DTOA.reindex(market_df.index,method='ffill')
    DTOADIF = DTOADIF.reindex(market_df.index,method='ffill')
    #负债权益比
    BLEV = TD/finance_df['total_shareholders_equity_including_MI']
    BLEV = BLEV.reindex(market_df.index, method='ffill')
    #ROS 销售利润率
    ROS = finance_df['deductedprofit']/finance_df['operating_revenue']
    ROS = ROS.reindex(market_df.index, method='ffill')
    # ROA 总资产利润率
    ROA = finance_df['deductedprofit']/finance_df['total_assets']
    ROA = ROA.reindex(market_df.index, method='ffill')
    # CF2A 经营现金流比总资产
    CF2A = finance_df['net_operating_cashflow']/finance_df['total_assets']
    CF2A = CF2A.reindex(market_df.index, method='ffill')
    # ROE 总资产利润率
    ROE = finance_df['deductedprofit']/finance_df['total_shareholders_equity_including_MI']
    ROE = ROE.reindex(market_df.index, method='ffill')
    FINANCE = pd.concat([MLEV,DTOA,DTOADIF,BLEV,ROS,ROA,CF2A,ROE],axis=1)
    FINANCE.columns = ['MLEV','DTOA','DTOADIF','BLEV','ROS','ROA','CF2A','ROE']
    #行业指标
    code_ind = comp_industry_dic[codenum[0]]
    #做虚拟变量
    all_ind = pd.DataFrame(np.zeros([len(market_df.index),len(ind_class)]),index=market_df.index,columns=ind_class)
    dummy = pd.DataFrame(np.ones([len(market_df.index),1]),index=market_df.index,columns=[code_ind])
    all_ind.update(dummy)
    all_factors = pd.concat([SIZE,factor_betas,MOM,BTM,TO,PORFIT,FINANCE,all_ind],axis=1)
    all_factors.to_csv('G:\\factors\\'+ codenum[0]+'.csv')
    mylog().info(codenum[0]+' \'s factors has been made !')
    return
for codenum in all_code:
    try:
        factormaker(codenum)
    except:
        mylog().error('Some error happend in '+ codenum[0] + ' !')










