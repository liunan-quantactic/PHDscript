from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from sqlalchemy import create_engine
import statsmodels.api as sm
import sys
from scipy.stats import ttest_1samp,zscore,t,f
sys.path.append("G:\\moudles")
defaultpath='G:\\'
def mylog(logpath=defaultpath):
    '''this moduel is to avoid printing log repeatly'''
    import logging.handlers
    logger = logging.getLogger('DBA')
    if 'log' not in os.listdir(logpath):
        final_path=os.mkdir(logpath+'log')
    else:
        final_path =logpath+'log'
    final_log=final_path+'\\log_'+datetime.datetime.now().strftime('%Y%m%d')+'.log'
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler1 = logging.FileHandler(final_log)
        handler1.setLevel(logging.DEBUG)
        handler2 = logging.StreamHandler()
        handler2.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger.addHandler(handler1)
        logger.addHandler(handler2)
    return logger

def sql_connector(username,password,server,port,schema):
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
    except:
        mylog().error('Fail to connect to SQL,please check your arguments ! ')

engine = sql_connector('root','7026155@Liu','127.0.0.1','3306','astocks')
#将所有有效的universe
universe = [x[0:9] for x in os.listdir('G:\\factors')]
num = len(universe)
i = 0
#数据描述阶段将所有的数据都按照月份重新排列保存到一个文件中:
Mdf = pd.DataFrame()
for codenum in universe:
    i = i+1 #count
    #数据调整的对应关系为本月月低factor对应下个月整月的累计收益
    fac_df = pd.read_csv("G:\\factors\\"+codenum+'.csv',index_col=[0],parse_dates=True)
    chg_df = pd.read_csv("G:\\backup\\market\\" + codenum + '.csv', index_col=[0], parse_dates=True)
    accum_return = np.cumprod(1+chg_df['chg']/100)#累计收益
    accum_return = accum_return.to_frame()
    accum_return_OM = accum_return.resample('M').last()#截至每个月低的累计收益
    return_per_mon = accum_return_OM.pct_change(1)#本月的涨幅
    adjust_fac_df = fac_df.resample('M').last()#因子取每个月月低的数据
    adjust_fac_df.index = adjust_fac_df.index+pd.tseries.offsets.MonthEnd(1)
    adjust_fac_df = adjust_fac_df.reindex(return_per_mon.index)#每个月的收益是由上个月低的因子暴露决定的
    adjust_fac_df.insert(0,'chg',return_per_mon['chg'].values)
    adjust_fac_df.insert(0, 'codenum', np.array([codenum]*len(return_per_mon['chg'])))#把标的代码写进去
    try:
        adjust_fac_df.to_csv("G:\\mfactor\\"+codenum+".csv")
        Mdf = pd.concat([Mdf,adjust_fac_df])#将新生成的数据写进文件
        mylog().info(codenum + ' \'factor data has been writing to file,process: '+str(i)+'/'+str(num))
    except:
        mylog().error(codenum + ' \'factor data has been wrong !,process: '+str(i)+'/'+str(num))
    Mdf.to_csv("G:\\all_factor_month.csv")


#取得研究数据
obj_df = pd.read_csv("G:\\all_factor_month.csv",index_col=0,parse_dates=True)
#调整时间
obj_df = obj_df[obj_df.index > '2007-01-01']

def factordescribe(df, facname, n=3):
        '''the argument of df should include factors and  change of price and weight vector if required'''
        '''df--the original data
           facname--factor name type:str
           n--the group number
           weight--the weight of component of portfolio'''
        '''step 2 按照因子大小分位数分组'''
        # 得所需要的时间
        period = np.unique(df.index)
        all_market_return_cv = []#计算市场整体的收益情况，以计算alpha以及进行T检验
        all_market_return_ev = []
        all_port_return_cv = []
        all_port_return_ev = []
        for t in period:
            tdf = df.loc[t]
            all_weight = tdf['total_EV']/tdf['total_EV'].sum()
            all_market_return_cv.append(np.sum(tdf['chg'] * all_weight))
            all_market_return_ev.append(tdf['chg'].mean())
            #因子分组是从小到大按照quantile进行分组 P3>P2>P1
            if facname != 'CD2EV':
                classes = np.array(pd.qcut(tdf[facname],n, ['P1', 'P2', 'P3']).tolist())
            else:#有部分数据如分红，区分度很小无法用quantile的方式分组，只能按照正常的bins均分
                classes = np.array(pd.cut(tdf[facname], n, labels=['P1', 'P2', 'P3']).tolist())
            tdf.insert(0, 'class', classes)
            gdf = tdf.groupby('class')
            T_port_return_cv = []
            T_port_return_ev = []
            for key in ['P1', 'P2', 'P3']:  # 不能用df.group.keys,顺序会错
                port = gdf.get_group(key)
                cap = port['total_EV']
                port_cap = cap.sum()
                weight = cap / port_cap
                port_return_cv = np.sum(port['chg'] * weight)
                port_return_ev = np.mean(port['chg'])
                T_port_return_cv.append(port_return_cv) #得到每一个时间区间的n个组合收益
                T_port_return_ev.append(port_return_ev)
            all_port_return_cv.append(T_port_return_cv)      #得到全部时间区间上的收益
            all_port_return_ev.append(T_port_return_ev)
        all_port_return_cv = pd.DataFrame(all_port_return_cv, index=period,columns=['P1', 'P2', 'P3'])
        all_port_return_ev = pd.DataFrame(all_port_return_ev, index=period, columns=['P1', 'P2', 'P3'])
        all_port_return_cv.insert(3, 'P3-P1', all_port_return_cv['P3'] - all_port_return_cv['P1'])
        all_port_return_ev.insert(3, 'P3-P1', all_port_return_ev['P3'] - all_port_return_ev['P1'])
        all_market_return_cv = pd.DataFrame(all_market_return_cv, index=period, columns=['all'])
        all_market_return_ev = pd.DataFrame(all_market_return_ev, index=period, columns=['all'])
        accum_all_port_return_cv = (1+all_port_return_cv/100).cumprod()      #全部收益的累计收益
        accum_all_port_return_ev = (1 + all_port_return_ev / 100).cumprod()
        accum_all_market_return_cv = (1+all_market_return_cv/100).cumprod()
        accum_all_market_return_ev = (1 + all_market_return_ev / 100).cumprod()
        fig = plt.figure(figsize=(10, 6), dpi=120)
        ax_01 = fig.add_subplot(2, 1, 1)
        title_str = 'Performance of '+facname+' factor portfolios (CW)'
        accum_all_port_return_cv[['P1', 'P2', 'P3']].plot(ax=ax_01,style=['g:.', 'b-.x', 'r--^'],fontsize=10)
        ax_01.set_title(title_str,fontsize=12)
        ax_02 = fig.add_subplot(2, 1, 2)
        accum_all_port_return_cv[['P3-P1']].plot(ax=ax_02, color='orange', rot=0,fontsize=10)
        ax_02.set_title('Excess Return of Factor Mimicing Portfolio(CW)',fontsize=12)
        plt.subplots_adjust(hspace=0.2)
        plt.savefig('G:\\output\\factorperformance\\'+facname+'_cw.png')
        fig = plt.figure(figsize=(10, 6), dpi=120)
        ax_01 = fig.add_subplot(2, 1, 1)
        title_str = 'Performance of '+facname+' factor portfolios (EW)'
        accum_all_port_return_ev[['P1', 'P2', 'P3']].plot(ax=ax_01,style=['g:.', 'b-.x', 'r--^'],fontsize=10)
        plt.title(title_str,fontsize=12)
        ax_02 = fig.add_subplot(2, 1, 2)
        accum_all_port_return_ev[['P3-P1']].plot(ax=ax_02, color='orange', rot=0,fontsize=10)
        ax_02.set_title('Excess Return of Factor Mimicing Portfolio(EW)',fontsize=12)
        plt.subplots_adjust(hspace=0.2)
        plt.savefig('G:\\output\\factorperformance\\'+facname+'_ew.png')
        #平均年化收益进行对比
        acc_port_pct_chg_cv = accum_all_port_return_cv.pct_change(12) #每12个月的滚动收益率并去除空值
        acc_port_pct_chg_cv.dropna(inplace=True)
        acc_port_pct_chg_ev = accum_all_port_return_ev.pct_change(12)
        acc_port_pct_chg_ev.dropna(inplace=True)
        acc_market_pct_chg_cv = accum_all_market_return_cv.pct_change(12)
        acc_market_pct_chg_cv.dropna(inplace=True)
        acc_market_pct_chg_ev = accum_all_market_return_ev.pct_change(12)
        acc_market_pct_chg_ev.dropna(inplace=True)
        acc_port_pct_chg_mean_cv = acc_port_pct_chg_cv.mean() #平均年化收益率
        acc_port_pct_chg_mean_ev = acc_port_pct_chg_ev.mean()
        acc_market_pct_chg_mean_cv = acc_market_pct_chg_cv.mean()
        acc_market_pct_chg_mean_ev = acc_market_pct_chg_ev.mean()
        alpha_cv = acc_port_pct_chg_mean_cv - acc_market_pct_chg_mean_cv.values #平均年化收益的alpha
        alpha_ev = acc_port_pct_chg_mean_ev-acc_market_pct_chg_mean_ev.values
        acc_port_pct_chg_std_cv = acc_port_pct_chg_cv.std() #平均年化波动率
        acc_port_pct_chg_std_ev = acc_port_pct_chg_ev.std()
        ttest_cv = ttest_1samp(acc_port_pct_chg_cv,acc_market_pct_chg_mean_cv.values)
        ttest_ev = ttest_1samp(acc_port_pct_chg_ev,acc_market_pct_chg_mean_ev.values)
        sharp_cv = acc_port_pct_chg_mean_cv/acc_port_pct_chg_std_cv
        sharp_ev = acc_port_pct_chg_mean_ev/acc_port_pct_chg_std_ev
       #汇总
        all_mean = np.concatenate([acc_port_pct_chg_mean_cv.values,acc_port_pct_chg_mean_ev.values])
        all_std=np.concatenate([acc_port_pct_chg_std_cv.values,acc_port_pct_chg_std_ev.values])
        all_ttest = np.concatenate([ttest_cv[0], ttest_ev[0]])
        all_pvalue=np.concatenate([ttest_cv[1], ttest_ev[1]])
        all_alpha = np.concatenate([alpha_cv.values, alpha_ev.values])
        all_sharp = np.concatenate([sharp_cv.values, sharp_ev.values])
        describe = np.stack([all_mean,all_std,all_ttest,all_pvalue,all_alpha,all_sharp],axis=0)
        describe = pd.DataFrame(describe,index=['mean','std','t_stat','p_value','alpha','sharp'],
                              columns=['P1_CW','P2_CW','P3_CW','P3-P1_CW','P1_EW','P2_EW','P3_EW','P3-P1_EW'])
        describe.to_csv('G:\\output\\factorperformance\\'+facname+'_factordescribe.csv')
        return
#得到total_EV的整体描述
all_facname = ['total_EV','free_EV','float_EV','beta','MOMOM','MOM3T2','MOM12T2',
               'BTM','STOM','STOQ','STOY','CETOP','EBITDA2EV','PETTM','GPOY','GPO3Y',
               'CD2EV','MLEV','DTOA','DTOADIF','BLEV','ROS','ROA','CF2A','ROE']
for facname in all_facname:
    try:
        factordescribe(obj_df, facname)
        mylog().info(facname+' \'s description is over ! ')
    except:
        mylog().error(facname + ' \'s description crashed ! ')


#Fama-Macbeth method
#取得研究数据
obj_df = pd.read_csv("G:\\all_factor_month.csv",index_col=0,parse_dates=True)
#调整时间
obj_df = obj_df[obj_df.index > '2007-01-01']
all_facname = ['total_EV','free_EV','float_EV','beta','MOMOM','MOM3T2','MOM12T2',
               'BTM','STOM','STOQ','STOY','CETOP','EBITDA2EV','PETTM','GPOY','GPO3Y',
               'CD2EV','MLEV','DTOA','DTOADIF','BLEV','ROS','ROA','CF2A','ROE']


def FamaMac(df, facname, method=None, categ='style',reg='OLS'):
    '''categ--factor category:style factor or system factor
       method--option log or none'''
    if reg=='OLS':
        if categ == 'style':
            period = np.unique(df.index)
            spot_beta = []
            spot_std = []
            for t in period:
                tdf = df.loc[t]
                tdf.dropna(inplace=True)
                tdf = tdf.replace([np.inf,-np.inf],0)
                CS_ret = tdf['chg']
                CS_Weight=tdf['total_EV']/tdf['total_EV'].sum()
                W=np.diag(CS_Weight.values)
                CS_fac = tdf[facname]
                if method == 'log':
                    CS_fac = np.log(CS_fac)
                CS_fac = zscore(CS_fac)  # 得到标准分数
                CS_fac = sm.add_constant(CS_fac)
                model = sm.GLS(CS_ret, CS_fac,W).fit()
                spot_beta.append(model.params[1])
                spot_std.append(model.bse[1])
                print(str(t) + '.....' + facname)
            T_reg = pd.DataFrame(np.stack([spot_beta, spot_std], axis=1), index=period, columns=['beta', 'std'])
            T_reg = T_reg.resample('Y').mean()
            T_reg.to_csv('G:\\factor_beta\\FM_' + facname + '.csv')
            beta = T_reg['beta'].mean()
            sigma = T_reg['std'].mean()
            tvalue = beta / sigma
            pd.DataFrame(np.stack([beta, sigma, tvalue]),index=['beta', 'sigma', 'tvalue']).to_csv('G:\\factor_beta\\FM_' + facname + '_sum.csv')
    elif reg=='WLS':
        if categ == 'style':
            period = np.unique(df.index)
            spot_beta = []
            spot_std = []
            for t in period:
                tdf = df.loc[t]
                tdf.dropna(inplace=True)
                tdf = tdf.replace([np.inf, -np.inf], 0)
                CS_ret = tdf['chg']
                CS_Weight = tdf['total_EV'] / tdf['total_EV'].sum()
                W = np.diag(CS_Weight.values)
                CS_fac = tdf[facname]
                if method == 'log':
                    CS_fac = np.log(CS_fac)
                CS_fac = zscore(CS_fac)  # 得到标准分数
                CS_fac = sm.add_constant(CS_fac)
                model = sm.GLS(CS_ret, CS_fac, W).fit()
                spot_beta.append(model.params[1])
                spot_std.append(model.bse[1])
                print(str(t) + '.....' + facname)
            T_reg = pd.DataFrame(np.stack([spot_beta, spot_std], axis=1), index=period, columns=['beta', 'std'])
            T_reg = T_reg.resample('Y').mean()
            T_reg.to_csv('G:\\factor_beta_gls\\FM_' + facname + '.csv')
            beta = T_reg['beta'].mean()
            sigma = T_reg['std'].mean()
            tvalue = beta / sigma
            pd.DataFrame(np.stack([beta, sigma, tvalue]), index=['beta', 'sigma', 'tvalue']).to_csv(
                'G:\\factor_beta_gls\\FM_' + facname + '_sum.csv')
    elif reg == 'GMM':
        # 求出因子残差进一步求因子之间的相关性
        universe = np.unique(df['codenum'])
        period = np.unique(df.index)
        all_residual = []
        all_code = []
        # 对每一个资产的相关因子求不带截距的回归
        for codenum in universe:
            code_df = df[df['codenum'] == codenum]
            if facname == 'market':
                fac = accum_index_return_chg
            else:
                fac = code_df[facname]
            ret = code_df['chg']
            fac = fac.reindex(ret.index)
            fac.fillna(method='ffill', inplace=True)
            fac.fillna(0, inplace=True)

            fac =sm.add_constant(fac)
            fac = fac.replace([np.inf, -np.inf], 0)#对于某些比率形式的因子有可能因为分母为0而出现无穷项，这种值设置为0
            ret = ret.reindex(fac.index)
            ret = ret.fillna(0)  # 对于上市日期晚于起始日期的个股填充空值
            model = sm.OLS(ret, fac).fit()
            if np.isnan(model.rsquared_adj):
                continue
            else:
                model = sm.OLS(ret, fac).fit()
                all_code.append(codenum)
                residual = model.resid.reindex(period)
                residual.fillna(0, inplace=True)
                all_residual.append(residual.values.tolist())
                print codenum
        GRS_resid = pd.DataFrame(np.array(all_residual).T, columns=all_code)
        GRS_resid.to_csv('G:\\factor_beta_gmm\\'+facname+'_resid.csv')
        #计算出个股之间残差的协方差矩阵#
        V = GRS_resid.cov()
        if np.all(np.linalg.eigvals(V.values)>=0)==False:
            V=pd.DataFrame(np.diag(np.diag(V)),index=V.index,columns=V.columns)
        if categ == 'style':
            period = np.unique(df.index)
            spot_beta = []
            spot_std = []
            for t in period:
                tdf = df.loc[t]
                tdf.dropna(inplace=True)
                T_V=V[tdf['codenum']].loc[tdf['codenum']]
                tdf = tdf.replace([np.inf, -np.inf], 0)
                CS_ret = tdf['chg']
                CS_Weight = tdf['total_EV'] / tdf['total_EV'].sum()
                W = np.diag(CS_Weight.values)
                T_W=T_V*W
                CS_fac = tdf[facname]
                if method == 'log':
                    CS_fac = np.log(CS_fac)
                CS_fac = zscore(CS_fac)  # 得到标准分数
                CS_fac = sm.add_constant(CS_fac)
                model = sm.GLS(CS_ret, CS_fac, T_W.values).fit()
                spot_beta.append(model.params[1])
                spot_std.append(model.bse[1])
                print(str(t) + '.....' + facname)
            T_reg = pd.DataFrame(np.stack([spot_beta, spot_std], axis=1), index=period, columns=['beta', 'std'])
            T_reg = T_reg.resample('Y').mean()
            T_reg.to_csv('G:\\factor_beta_gmm\\FM_' + facname + '.csv')
            beta = T_reg['beta'].mean()
            sigma = T_reg['std'].mean()
            tvalue = beta / sigma
            pd.DataFrame(np.stack([beta, sigma, tvalue]), index=['beta', 'sigma', 'tvalue']).to_csv(
                'G:\\factor_beta_gmm\\FM_' + facname + '_sum.csv')
        '''beta0 = np.dot(np.linalg.inv(np.dot(np.dot(X.T, np.kron(np.kron(Z, W), Z.T)), X)),
                       np.dot(np.dot(X.T, np.dot(np.dot(Z, W), Z.T)), y))
        residual1 = np.mat(y1 - np.dot(X, beta0)).T
        mom_cond = Z
        sigma = np.dot(np.dot(Z.T, float(np.dot(residual.T, residual) / (t - 2))), Z)
        W_new = np.array(np.linalg.inv(sigma))
        beta1 = np.dot(np.linalg.inv(np.dot(np.dot(X.T, np.dot(np.dot(Z, W_new), Z.T)), X)),
                       np.dot(np.dot(X.T, np.dot(np.dot(Z, W_new), Z.T)), y))
        F = np.linalg.inv(np.dot(np.dot(X.T, np.dot(np.dot(Z, W_new), Z.T)), X))
        var_beta1 = np.dot(
            np.dot(F, np.dot(np.dot(X.T, np.dot(np.dot(Z, np.dot(np.dot(W_new, sigma), W_new.T)), Z.T)), X)), F)
        std_beta1 = np.sqrt(np.diag(var_beta1))
        t_stat = beta1 / std_beta1
        p_value = sta.t.sf(t_stat, t - 2)'''
    return
for facname in all_facname:#按照因子进loop
    if facname in ['total_EV','free_EV','float_EV']:
        method = 'log'
    else:
        method = None
    FamaMac(obj_df,facname,method=method,reg='GMM')




for facname in all_facname:
    #1、先求出每个股票在因子上面的beta,将所有的信息放到一个表里去
    #数据描述阶段将所有的数据都按照月份重新排列保存到一个文件中:
    def ts_reg(codenum, facname, freg='M', method=None):
        # 以每个月作为滚动，回归窗口为每次3年，36个月，或者12个季
        df = pd.read_csv("G:\\mfactor\\" + codenum + ".csv", index_col=0, parse_dates=True)  # 得到的已经是月数据
        df = df[df['STOM']!=0]
        total_EV = df['total_EV']
        chg = df['chg']
        fac = df[facname]
        fac.dropna(inplace=True)
        if method == 'log':
            fac = np.log(fac)
        else:
            fac = fac
        ret = df['chg'].reindex(fac.index)
        total_EV = total_EV.reindex(fac.index)
        chg = chg.reindex(fac.index)
        if freg == 'M':
            adjust_index=fac.index
        elif freg == 'Q':#先出来，还没功能
            adjust_index = fac.index
        params = []
        sigmas = []
        tvalues = []
        interval = []
        resid_mean = []
        resid_std = []
        spot_fac = []
        spot_EV= []
        interval_chg = []
        fac = sm.add_constant(fac)#此处为含有截距项的
        if len(fac) <= 35:#对于上市不满3年的企业，就只能选择使用其所能提供的数据进行回归
            model = sm.OLS(ret, fac).fit()
            beta = model.params.values  # 3年的beta（可以理解成因子收益），频率仍然为月
            sigma = model.bse  # 3年个月系数的标准差，可以选择robust
            tvalue = model.tvalues  # 3年个月系数的t
            residual_mean = np.mean(model.resid)  # 3年系数对应的残差均值
            residual_std = np.std(model.resid)  # 3年月系数对应的残差均值
            params.append(beta)
            sigmas.append(sigma)
            tvalues.append(tvalue)
            resid_mean.append(residual_mean)
            resid_std.append(residual_std)
            interval.append(adjust_index[-1])
            spot_fac.append(fac.values[-1][1])
            spot_EV.append(total_EV.values[-1])
            interval_chg.append(chg.values[-1])
            beta_df = pd.DataFrame(np.column_stack([spot_fac,spot_EV,interval_chg,params, sigmas, tvalues, resid_mean, resid_std]),
                                   index=pd.to_datetime(interval, format='%Y%m%'),
                                   columns=['facname','total_EV','chg','incpt', 'beta', 'incpt_std', 'beta_std', 't_incpt', 't_beta', 'res_avrg',
                                            'res_std'])
            beta_df.insert(0, 'codenum', np.array([codenum]))
            beta_df.dropna(inplace=True)
            try:
                beta_df.to_csv('G:\\factor_beta\\'+facname+'\\'+codenum+'.csv')
                mylog().info(codenum+'\'s '+facname+ ' TS regression is completed !')
            except:
                mylog().error(codenum + '\'s ' + facname + ' TS regress is wrong !')
        else:
            for i in range(len(adjust_index)):#i 是从0开始的 长度是36，max range is 35
                if i - 35 >= 0:#当i=35时已经是第36个数字
                    rolling_ret = ret[adjust_index[i - 35]:adjust_index[i]]
                    rolling_fac = fac[adjust_index[i - 35]:adjust_index[i]]
                    model = sm.OLS(rolling_ret, rolling_fac).fit()
                    beta = model.params.values  # 3年的beta（可以理解成因子收益），频率仍然为月
                    sigma = model.bse  # 3年个月系数的标准差，可以选择robust
                    tvalue = model.tvalues  # 3年个月系数的t
                    residual_mean = np.mean(model.resid)  # 3年系数对应的残差均值
                    residual_std = np.std(model.resid)  # 3年月系数对应的残差均值
                    params.append(beta)
                    sigmas.append(sigma)
                    tvalues.append(tvalue)
                    resid_mean.append(residual_mean)
                    resid_std.append(residual_std)
                    interval.append(adjust_index[i])
                    spot_fac.append(fac.values[i][1])
                    spot_EV.append(total_EV.values[i])
                    interval_chg.append(chg.values[i])
            beta_df = pd.DataFrame(np.column_stack([spot_fac,spot_EV,interval_chg,params, sigmas, tvalues, resid_mean, resid_std]),
                                   index=pd.to_datetime(interval, format='%Y%m%'),
                                   columns=['facname','total_EV','chg','incpt', 'beta', 'incpt_std', 'beta_std', 't_incpt', 't_beta', 'res_avrg',
                                            'res_std'])
            beta_df.insert(0, 'codenum', np.array([codenum] * len(beta_df.index)))
            beta_df.dropna(inplace=True)
            try:
                beta_df.to_csv('G:\\factor_beta\\'+facname+'\\'+codenum+'.csv')
                mylog().info(codenum+'\'s '+facname+ ' TS regression is completed !')
            except:
                mylog().error(codenum + '\'s ' + facname + ' TS regress is wrong !')
        return beta_df
        # 注：reindex的时候开始阶段有NA值
    #将所有结果放入一个文件中便于分组
    TS_df = pd.DataFrame()
    if facname in ['total_EV','free_EV','float_EV']:
        method = 'log'
    else:
        method = None
    for codenum in universe:
        #数据调整的对应关系为本月月低                                                                                                                                                   factor对应下个月整月的累计收益
        try:
            itsg_df = ts_reg(codenum, facname, freg='M', method=method) #individual time series regress
            TS_df = pd.concat([TS_df, itsg_df])
        except:
            mylog().error('Some wrong happend when time regress in '+codenum)
    TS_df.to_csv("G:\\factor_beta\\"+facname+".csv")
    #2、按照因子大小分为10组，求出每组平均的beta
    #分组，求出每组因子的平均beta以及加权收益率，index仍然为时间
    T_port_lamb = []
    T_port_std = []
    T_port_beta = []
    TS_df = TS_df[TS_df.index >= '2007-01-01']
    period = np.unique(TS_df.index)
    for t in period:
        tdf = TS_df.loc[t]
        t_num = 10
        if facname != 'CD2EV':
            #
            classes = np.array(pd.qcut(tdf['facname'], t_num, ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']).tolist())
        else:  # 有部分数据如分红，区分度很小无法用quantile的方式分组，只能按照正常的bins均分
            classes = np.array(pd.cut(tdf[facname], t_num,
                                      labels=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']).tolist())
        tdf.insert(0, 'class', classes)
        gdf = tdf.groupby('class')
        CS_port_return = []
        CS_port_beta = []
        for key in ['P1', 'P2', 'P3','P4','P5','P6','P7','P8','P9','P10']:  # 不能用df.group.keys,顺序会错
            port = gdf.get_group(key)
            CS_fac = port['facname']#组合的
            cap = port['total_EV']#组合成分的市值，用来求权重
            port_cap = cap.sum()
            weight = cap / port_cap   #求出权重
            port_return_cv = np.sum(port['chg'] * weight)   #组合的加权收益率
            port_beta = np.mean(port['beta'])               #组合的平均beta
            CS_port_return.append(port_return_cv)
            CS_port_beta.append(port_beta)
        CS_port_regressor = sm.add_constant(CS_port_beta)
        CS_reg = sm.OLS(CS_port_return,CS_port_regressor).fit()#cross-sectional regression
        T_port_lamb.append(CS_reg.params[1])
        T_port_std.append(CS_reg.bse[1])
        T_port_beta.append(CS_port_beta)
    CS_reg_df = pd.DataFrame(np.stack([T_port_lamb,T_port_std],axis=1),index=period,columns=['lambda','std'])
    Y_CS_reg_df = CS_reg_df.resample('Y').mean()
    Y_CS_reg_df.to_csv("G:\\factor_beta\\"+facname+"_lambda.csv")
    all_port_beta = pd.DataFrame(np.array(T_port_beta),index=period,columns=['P1', 'P2', 'P3','P4','P5','P6','P7','P8','P9','P10'])
    fig = plt.figure(figsize=(10, 6), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    title_str = 'Different TS regression coefficent in representative '+facname+' portfolio'
    all_port_beta[['P1', 'P5', 'P10']].plot(ax=ax,style=['g:.', 'b-.x', 'r--^'],fontsize=10)
    Y_all_port_beta=all_port_beta.resample('Y').mean()
    Y_all_port_beta.to_csv("G:\\factor_beta\\"+facname+"_beta.csv")
    ax.set_title(title_str,fontsize=14)
    plt.savefig("G:\\factor_beta\\"+facname+".png")
    tvalue = CS_reg_df['lambda'].mean()/CS_reg_df['std'].mean()
    pvalue = ss.t.sf(tvalue,len(period)-1)
    t_df = pd.DataFrame(np.stack([tvalue,pvalue]))
    t_df.to_csv("G:\\factor_beta\\"+facname+"_t.csv")


#GMM方法改进的Fama-Macbeth 方法




##GRS test
from quantactic import sql_connector,initialdata,mylog
user = 'root'
pw = '7026155@Liu'
h = '127.0.0.1'
p = 3306
sch = 'astocks'
engine = sql_connector(user,pw,h,p,sch)
index_query = 'select td,chg from astocks.indexprice where codenum=\''+'000300.SH'+'\';'
index_df = pd.read_sql(index_query,con=engine,index_col='td')
index_df.index = pd.to_datetime(index_df.index,format='%Y%m%d')
accum_index_return = np.cumprod(1+index_df['chg']/100)
accum_index_return = accum_index_return.to_frame()
accum_index_return_OM = accum_index_return.resample('M').last()  # 截至每个月底的累计收益
accum_index_return_chg = accum_index_return_OM.pct_change(1)  # 本月的涨幅

#accum_index_return_chg.dropna(inplace=True)#为归一化处理做准备
#new_index=accum_index_return_chg.index
#accum_index_return_chg =zscore(accum_index_return_chg['chg'].values)
#accum_index_return_chg=pd.DataFrame(accum_index_return_chg,index=new_index,columns=['chg'])
def GRStest(df,facname):
    '''GRS test 需要返回所有GRS F 统计量，平均alpha，平均的时间序列回归的R square 横截面的R square'''
    #先进行时间序列上的回归
    universe = np.unique(df['codenum'])
    period=np.unique(df.index)
    all_ts_reg_R = []
    all_ts_reg_alpha = []
    all_code = []
    all_residual = []
    #对每一个资产的相关因子求不带截距的回归
    for codenum in universe:
        code_df = df[df['codenum'] == codenum]
        if facname == 'market':
            fac = accum_index_return_chg
        else:
            fac = code_df[facname]
        ret = code_df['chg']
        fac = fac.reindex(ret.index)
        fac.fillna(method='ffill',inplace=True)
        fac.fillna(0,inplace=True)
        ret = ret.reindex(fac.index)
        ret = ret.fillna(0)#对于上市日期晚于起始日期的个股填充空值
        model = sm.OLS(ret,fac).fit()
        if np.isnan(model.rsquared_adj):
            continue
        else:
            ts_rsquare = model.rsquared_adj#使用调整过的Rsquare
            all_ts_reg_R.append(ts_rsquare)
            const_fac = sm.add_constant(fac)
            model = sm.OLS(ret,const_fac).fit()
            ts_alpha = model.params.loc['const']
            all_ts_reg_alpha.append(ts_alpha)
            all_code.append(codenum)
            #model.resid = pd.DataFrame(zscore(model.resid),index=model.resid.index)#归一化处理后找相关性
            residual = model.resid.reindex(period)
            residual.fillna(0,inplace=True)
            all_residual.append(residual.values.tolist())
            print codenum
    GRS_TS = pd.DataFrame(np.stack([all_ts_reg_alpha,all_ts_reg_R],axis=1),index=all_code,columns=['alpha','R_adj'])
    GRS_resid = pd.DataFrame(np.array(all_residual).T,columns=all_code)
    GRS_TS.dropna(inplace=True)
    mean_ts_reg_R = GRS_TS['R_adj'].mean()
    mean_all_ts_reg_alpha=GRS_TS['alpha'].mean()
    x=[mean_all_ts_reg_alpha,mean_ts_reg_R]
    GRS_TS.append(pd.DataFrame([x],index=['average'],columns=['alpha','R_adj']))
    GRS_TS.to_csv('G:\\GRS_TS.csv')
    GRS_resid.to_csv('G:\\GRS_resid.csv')


    #横截面回归
    #不加权处理可以更好的反映因子组合对于收益的解释能力
    period = np.unique(df.index) #得到全部的时间数据
    CS_ret = df['chg']
    accum_index_return_chg = accum_index_return_chg.reindex(period)
    if facname == 'market':
        CS_fac = accum_index_return_chg
    else:
        CS_fac = df[facname]
    #求出每个截面上收益和因子的均值
    CS_avg_ret = []
    CS_avg_fac = []
    for t in period:
        t_CS_ret = CS_ret.loc[t].mean()
        t_CS_fac = CS_fac.loc[t].fillna(0).values.mean(axis=0).tolist()
        CS_avg_ret.append(t_CS_ret)
        CS_avg_fac.append(t_CS_fac)
    CS_avg_fac=pd.DataFrame(CS_avg_fac,index=period,columns=[facname])
    CS_avg_fac_mark = pd.concat([CS_avg_fac,accum_index_return_chg],axis=1)
    CS_model = sm.OLS(CS_avg_ret,sm.add_constant(CS_avg_fac_mark)).fit()



    #计算特征



    # GRS test
    # 各只股票在时间序列残差上的协方差
    residual_cov = GRS_resid.cov()
    # 求出因子于因子之间的协方差
    factor_cov = CS_avg_fac.cov()
    inv_factor_cov = np.linalg.inv(factor_cov)
    A = 1+np.dot(np.dot(CS_avg_fac.mean().values,inv_factor_cov),CS_avg_fac.mean().values.T)
    B = np.dot(np.dot(all_ts_reg_alpha,np.linalg.inv(residual_cov)),all_ts_reg_alpha)
    N = len(all_code)
    T = len(GRS_resid.index)
    K = len(facname)
    Fstats=(T-K)*B/N/A
    pvalue=f.sf(Fstats,N,T-K)






'''OLS demo
n=3000
mean = [6,0] 
cov = np.eye(2,2)
V = np.random.multivariate_normal(mean,cov,n)
[X,error]=V.T

y=8+2.5*X+error
a_X=sm.add_constant(X)
modfit=sm.OLS(y,X).fit()
res=modfit.summary()
para=modfit.params
tvalues=modfit.tvalues'''

'''数据转化成为其它区间的数据,年数据，月数据，季数据'''
def accum_return(df,cycle):
    '''the cycle option include  month,season and year'''
    if cycle=='m' or cycle=='s':
        year_month=np.unique(np.array([x[:7] for x in mdf.index.astype('str')]))#get the %Y-%m format
        return_perm_list = []
        season=[]
        return_pers_list = []
        for ym in year_month:
            return_per_month=np.cumprod(1+df[ym]['chg'].values/100)[-1]-1
            return_perm_list.append(return_per_month)
            if ym[-2:]=='01':
                season1_return=[]
                season1_return.append(return_per_month)
            elif ym[-2:]=='02':
                season1_return.append(return_per_month)
            elif ym[-2:]=='03':
                season1_return.append(return_per_month)
                return_pers_list.append(np.cumprod(1+np.array(season1_return))[-1])
                season.append(ym[0:4]+'-I')
            elif ym[-2:]=='04':
                season2_return=[]
                season2_return.append(return_per_month)
            elif ym[-2:]=='05':
                season2_return.append(return_per_month)
            elif ym[-2:]=='06':
                season2_return.append(return_per_month)
                return_pers_list.append(np.cumprod(1+np.array(season2_return))[-1])
                season.append(ym[0:4]+'-II')
            elif ym[-2:]=='07':
                season3_return=[]
                season3_return.append(return_per_month)
            elif ym[-2:]=='08':
                season3_return.append(return_per_month)
            elif ym[-2:]=='09':
                season3_return.append(return_per_month)
                return_pers_list.append(np.cumprod(1+np.array(season3_return))[-1])
                season.append(ym[0:4]+'-III')
            elif ym[-2:]=='10':
                season4_return=[]
                season4_return.append(return_per_month)
            elif ym[-2:]=='11':
                season4_return.append(return_per_month)
            elif ym[-2:]=='12':
                season4_return.append(return_per_month)
                return_pers_list.append(np.cumprod(1+np.array(season4_return))[-1])
                season.append(ym[0:4]+'-IV')
        if cycle=='m':
            return_df = pd.DataFrame(return_perm_list, index=year_month, columns=['return'])
        else:
            return_df = pd.DataFrame(return_pers_list, index=season, columns=['return'])
    elif cycle=='y':
        return_pery_list = []
        year = np.unique(np.array([x[:4] for x in mdf.index.astype('str')]))  # get the %Y-%m format
        for y in year:
            return_per_year=np.cumprod(1+df[y]['chg'].values/100)[-1]-1
            return_pery_list.append(return_per_year)
        return_df = pd.DataFrame(return_pery_list, index=year, columns=['return'])
    return return_df

'''FM 回归的第一阶段回归，计算个股的beta，由于财务基本面数据一年只有4个数据，
用TTM数据调整季度,因此首次每次回归选用5年数据，且数据向前滚动进行回归'''

from quantactic import sql_connector,initialdata,mylog
user = 'root'
pw = '7026155@Liu'
h = '127.0.0.1'
p = 3306
sch = 'astocks'
engine = sql_connector(user,pw,h,p,sch)
index_query = 'select td,chg from astocks.indexprice where codenum=\''+'000985.CSI'+'\';'
index_df = pd.read_sql(index_query,con=engine,index_col='td')
index_df.index = pd.to_datetime(index_df.index,format='%Y%m%d')
accum_index_return = np.cumprod(1+index_df['chg']/100)
accum_index_return = accum_index_return.to_frame()
accum_index_return_OM = accum_index_return.resample('M').last()  # 截至每个月底的累计收益
accum_index_return_chg = accum_index_return_OM.pct_change(1)  # 本月的涨幅
accum_index_return_chg.dropna(inplace=True)#去掉空值
MKT_fac = pd.DataFrame(zscore(accum_index_return_chg), index=accum_index_return_chg.index,
                       columns=accum_index_return_chg.columns)

#多因子模型的检验
obj_df = pd.read_csv("G:\\all_factor_month.csv",index_col=0,parse_dates=True)
#调整时间
obj_df = obj_df[obj_df.index > '2007-01-01']
all_facname =  ['total_EV','free_EV','float_EV','beta','MOMOM','MOM3T2','MOM12T2',
               'BTM','STOM','STOQ','STOY','CETOP','EBITDA2EV','PETTM','GPOY','GPO3Y',
               'CD2EV','MLEV','DTOA','DTOADIF','BLEV','ROS','ROA','CF2A','ROE']

for facname in all_facname:
    #时序回归：求出平均的截距，和平均的R square
    universe = np.unique(obj_df['codenum'])
    period = np.unique(obj_df.index)
    all_ts_reg_R = []
    all_ts_reg_alpha = []
    all_code = []
    all_residual = []
    # 对每一个资产的相关因子求带截距的回归
    for codenum in universe:
        code_df = obj_df[obj_df['codenum'] == codenum]
        if facname == 'market':
            raw_fac =  MKT_fac
        #elif len(facname) >=2:
            #STY_fac = pd.DataFrame(zscore(code_df[facname]), index=code_df.index, columns=code_df[facname].columns)
            #raw_fac = pd.concat([MKT_fac, STY_fac], axis=1)
        else:
            STY_fac = pd.DataFrame(zscore(code_df[facname]), index=code_df.index, columns=[facname])
            raw_fac = STY_fac
        ret = code_df['chg']
        fac = raw_fac.reindex(ret.index)
        fac.fillna(method='ffill', inplace=True)
        fac.fillna(0, inplace=True)
        fac = fac.replace([np.inf, -np.inf], 0)
        ret = ret.reindex(fac.index)
        ret = ret.fillna(0)  # 对于上市日期晚于起始日期的个股填充空值
        fac = sm.add_constant(fac)
        model = sm.OLS(ret, fac).fit()
        if np.isnan(model.rsquared_adj):
            continue
        else:
            ts_rsquare = model.rsquared# 使用调整过的Rsquare
            if len(model.params) < len([facname])+1:#有时会出现数据异常
                continue
            else:
                all_ts_reg_R.append(ts_rsquare)
                ts_alpha = model.params.loc['const']
                all_ts_reg_alpha.append(np.abs(ts_alpha))
                all_code.append(codenum)
                print(codenum)
    TS_res = pd.DataFrame(np.stack([all_ts_reg_alpha, all_ts_reg_R], axis=1), index=all_code, columns=['alpha', 'R_adj'])
    TS_res.dropna(inplace=True)
    mean_ts_reg_R = TS_res['R_adj'].mean()
    std_ts_reg_R =TS_res['R_adj'].std()
    mean_all_ts_reg_alpha = TS_res['alpha'].mean()
    std_ts_reg_reg_alpha=TS_res['alpha'].std()
    x = [mean_all_ts_reg_alpha, std_ts_reg_reg_alpha]
    y=[mean_ts_reg_R,std_ts_reg_R]
    TS_res = TS_res.append(pd.DataFrame([x], index=['alpha'], columns=['alpha', 'R_adj']))
    TS_res = TS_res.append(pd.DataFrame([y], index=['R'], columns=['alpha', 'R_adj']))
    #做横截面回归
    # 求出每个截面上收益和因子的均值
    CS_ret = obj_df['chg']
    if facname == 'market':
        CS_fac = MKT_fac
    #elif len(facname) >= 2:
        #CS_fac = CS_fac.replace([np.inf, -np.inf], 0)
    else:
        CS_fac = obj_df[facname]
        CS_fac = CS_fac.replace([np.inf, -np.inf], 0)
    CS_avg_ret = []
    raw_CS_avg_fac = []
    for t in period:
        t_CS_ret = CS_ret.loc[t].mean()
        t_CS_fac = CS_fac.loc[t].fillna(0).values.mean(axis=0).tolist()
        CS_avg_ret.append(t_CS_ret)
        raw_CS_avg_fac.append(t_CS_fac)
    #if len(facname) >= 2:
        #raw_CS_avg_fac = pd.concat([pd.DataFrame(raw_CS_avg_fac,index=period), accum_index_return_chg], axis=1)
    #else:
    raw_CS_avg_fac = pd.DataFrame(raw_CS_avg_fac, index=period, columns=[facname])
    if facname!='market':
        CS_avg_fac = sm.add_constant(pd.DataFrame(zscore(raw_CS_avg_fac),index=raw_CS_avg_fac.index,columns=[facname]))
    else:
        CS_avg_fac=sm.add_constant(raw_CS_avg_fac)
    CS_model = sm.OLS(CS_avg_ret, CS_avg_fac).fit()
    f_cov = np.cov(CS_avg_fac[facname])
    beta = CS_model.params[1:].values
    egien = np.dot(np.dot(beta.T, f_cov), beta)
    ret_var = np.array(CS_avg_ret).var()
    eigenratio = egien / ret_var
    z = [CS_model.rsquared, eigenratio]
    TS_res =TS_res.append(pd.DataFrame([z], index=['CS_reg'], columns=['alpha', 'R_adj']))
    TS_res.to_csv("G:\\multifactor\\"+facname+"_TS.csv")
    mylog().info(facname+' is over !')
