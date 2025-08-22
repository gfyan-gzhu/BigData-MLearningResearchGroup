import baostock as bs
import pandas as pd
# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)
rs = bs.query_history_k_data_plus("sz.399001",
                                  "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                  start_date='2020-01-01', end_date='2022-01-01', frequency="d")
# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv("./data/399001sz.csv", index=False)
print(result)
# 登出系统
bs.logout()