import pandas as pd
import numpy as np
import datetime
import os
import statsmodels.api as sm

df=pd.read_csv("F:\\all_factor_month.csv",index_col=0,parse_dates=True)

'''GRS test 需要返回所有GRS F 统计量，平均alpha，平均的时间序列回归的R square 横截面的R square'''
# 先进行时间序列上的回归
universe = np.unique(df[df['codenum'].isna()==False].codenum)
period = np.unique(df.index)
all_ts_reg_R = []
all_ts_reg_alpha = []
all_code = []
all_residual = []
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
    ret = ret.reindex(fac.index)
    ret = ret.fillna(0)  # 对于上市日期晚于起始日期的个股填充空值
    model = sm.OLS(ret, fac).fit()
    if np.isnan(model.rsquared_adj):
        continue
    else:
        ts_rsquare = model.rsquared_adj  # 使用调整过的Rsquare
        all_ts_reg_R.append(ts_rsquare)
        const_fac = sm.add_constant(fac)
        model = sm.OLS(ret, const_fac).fit()
        ts_alpha = model.params.loc['const']
        all_ts_reg_alpha.append(ts_alpha)
        all_code.append(codenum)
        # model.resid = pd.DataFrame(zscore(model.resid),index=model.resid.index)#归一化处理后找相关性
        residual = model.resid.reindex(period)
        residual.fillna(0, inplace=True)
        all_residual.append(residual.values.tolist())
        print(codenum)
GRS_TS = pd.DataFrame(np.stack([all_ts_reg_alpha, all_ts_reg_R], axis=1), index=all_code, columns=['alpha', 'R_adj'])
GRS_resid = pd.DataFrame(np.array(all_residual).T, columns=all_code)
GRS_TS.dropna(inplace=True)
mean_ts_reg_R = GRS_TS['R_adj'].mean()
mean_all_ts_reg_alpha = GRS_TS['alpha'].mean()
x = [mean_all_ts_reg_alpha, mean_ts_reg_R]
GRS_TS.append(pd.DataFrame([x], index=['average'], columns=['alpha', 'R_adj']))
GRS_TS.to_csv('G:\\GRS_TS.csv')
GRS_resid.to_csv('G:\\GRS_resid.csv')

# 横截面回归
# 不加权处理可以更好的反映因子组合对于收益的解释能力
period = np.unique(df.index)  # 得到全部的时间数据
CS_ret = df['chg']
accum_index_return_chg = accum_index_return_chg.reindex(period)
if facname == 'market':
    CS_fac = accum_index_return_chg
else:
    CS_fac = df[facname]
# 求出每个截面上收益和因子的均值
CS_avg_ret = []
CS_avg_fac = []
for t in period:
    t_CS_ret = CS_ret.loc[t].mean()
    t_CS_fac = CS_fac.loc[t].fillna(0).values.mean(axis=0).tolist()
    CS_avg_ret.append(t_CS_ret)
    CS_avg_fac.append(t_CS_fac)
CS_avg_fac = pd.DataFrame(CS_avg_fac, index=period, columns=[facname])
CS_avg_fac_mark = pd.concat([CS_avg_fac, accum_index_return_chg], axis=1)
CS_model = sm.OLS(CS_avg_ret, sm.add_constant(CS_avg_fac_mark)).fit()

