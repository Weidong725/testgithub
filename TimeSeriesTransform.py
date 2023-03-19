

import pandas as pd 
import numpy as np


class TimeSeriesTransform(object):


    def __init__(self):
        # 96时刻点，每隔15min取一次，T0000，T0015，...，T2345
        self.freq96 = ["T"+ "{:02d}".format(m) + "{:02d}".format(h) for m in range(0,24) for h in range(0,60,15)]
        # 48时刻点 ，每隔30min取一次，T0000，T0030，T0100，...，T2330
        self.freq48 = self.freq96[::2]
        # 24时刻点 ，每隔1h取一次，T0000，T0100，T0200，...，T2300
        self.freq24 = self.freq96[::4]


        self.num2freq = {
            96:self.freq96,
            48:self.freq48,
            24:self.freq24
        }

        self.__num2freq_minutes = {
            96: 15,
            48: 30,
            24: 60
        }
        self.__num2freq_minutes_str = {
            96: '15min',
            48: '30min',
            24: '1h'
        }
    

    def connectDB(self,user:str,password:str,host:str,port:int,sql:str,dbType:str)->pd.DataFrame:       
        """用于读取达梦7或MYSQL数据库中的数据并转换为DataFrame


        Args:
        ----------
            user (str):  数据库用户名
            password (str): 数据库密码
            host (str): 数据库host地址
            port (int): 数据库端口
            sql (str): 查询sql语句
            dbType (str): 数据库类型,可输入'mysql','dm7'

        Raises:
            ValueError: _description_

        Returns:
            pd.DataFrame: 
        """
        if dbType == 'dm7':
            import dmPython
            conn = dmPython.connect(user=user, password=password, host=host, port=port, autoCommit=True)
        elif dbType == 'mysql':
            import pymysql
            conn = pymysql.connect(user=user, password=password, host=host, port=port)
        else:
            raise ValueError('dbType can only be entered "mysql" or "dm7" !')

        cursor = conn.cursor()
        cursor.execute(sql)
        col_tmp = cursor.description
        col_name = []
        for i in range(len(col_tmp)):
            col_name.append(col_tmp[i][0].split(',')[0])
        res = cursor.fetchall()

        df = pd.DataFrame(res,columns=[str(i).upper() for i in col_name])
        return df

    def read_excel(self, path, sheet_name=None):
        """
        用于读取excel文件

        Parameters:
        ----------
            path:str
                excel文件路径
            sheet_name:
                需读取的excel中sheet的名称，默认为None
        Returns:
        ----------
            DataFrame
        """
        self.df = pd.read_excel(path, sheet_name=sheet_name, index_col=0)

        return self.df.copy()

    def read_csv(self, path):
        """
        用于读取csv文件
        Parameters:
        ----------
            path:str
                csv文件路径
        Returns:
        ----------
            DataFrame
        """
        self.df = pd.read_csv(path, index_col=0)
        return self.df.copy()
    
    def transLoad(self,df:pd.DataFrame,time_col = 'DATE',cityid_col = 'CITY_ID',city_id:int =None,caliber_id=None,isDelCitycol=True,isNa2Null=False):
        """
        用于处理负荷数据,将负荷数据转为日期+地市ID+96时刻负荷的形式
        
        Parameters:
        ----------
            df:Dataframe 输入数据
            time_col:str df中日期列的名称
            cityid_colname:str df中城市id列的名称
            city_id:int 需筛选的地市ID,默认为None输出所有地市
            caliber_id:int 需要删选的口径id，默认为None，输出所有口径数据
            isDelCitycol:bool 是否直接删除cityid_col
            isNa2Null:bool 是否将数据中的NaN转为null
        Returns:
        ----------
            DataFrame
        """
        # 将地市id、口径id转为int型
        df[[cityid_col,'CALIBER_ID']] = df[[cityid_col,'CALIBER_ID']].astype(int)
        # 将日期列转为时间格式，如超出时间范围则替换为NaT
        df[time_col] = pd.to_datetime(df[time_col], errors = 'coerce')

        if caliber_id:
            df = df[df['CALIBER_ID']==caliber_id]

        # 删除时间列为空的行
        df.dropna(subset=[time_col],inplace=True)

        # 剔除无用列
        df = df.drop(['ID','CALIBER_ID','CREATETIME','UPDATETIME','T2400'],axis=1)

        # 重置列名
        if len(df.columns) == 96+2:      
            df.columns = [time_col] + [cityid_col] + self.freq96
        elif len(df.columns) == 48+2:
            df.columns = [time_col] + [cityid_col] + self.freq48
        elif len(df.columns) == 24+2:
            df.columns = [time_col] + [cityid_col] + self.freq24
        
        # 将空值都转换为Nan
        df.fillna(np.nan,inplace=True)

        # 是否需要将Nan转换为null
        if isNa2Null:
            df.replace(np.nan,'null',inplace=True)
        # 筛选地市
        if city_id:
            df = df[df[cityid_col]==city_id]
        # 删除city_id列
        if isDelCitycol:
            df = df.drop(cityid_col,axis=1)
        
        return df
    
    def transWeather(self,df:pd.DataFrame,time_col = 'DATE',cityid_col = 'CITY_ID',city_id = None,isNa2Null=False,isDelCitycol=True,isStat=False):
        """
        用于处理气象数据,将气象数据转为日期+地市ID+96时刻负荷的形式
        
        Parameters:
        ----------
            df:Dataframe 输入数据
            time_col:str df中日期列的名称
            cityid_colname:str df中城市id列的名称
            city_id:int 需筛选的地市ID,默认为None输出所有地市
            isNa2Null:bool 是否将数据中的NaN转为null
            isDelCitycol:bool 是否直接将cityid_col这一列删除
            isStat:bool 是否生成温度最大值、最小值和均值列
        Returns:
        ----------
            DataFrame
        """
        # 将地市id转为int型
        df[cityid_col] = df[cityid_col].astype(int)
        # 将日期列转为时间格式，如超出时间范围则替换为NaT
        df[time_col] = pd.to_datetime(df[time_col], errors = 'coerce')
        # 删除时间列为空的行
        df.dropna(subset=[time_col],inplace=True)

        # 剔除无用列
        df = df.drop(['ID','TYPE','CREATETIME','UPDATETIME','T2400'],axis=1)

        # 重置列名
        if len(df.columns) == 96+2:      
            df.columns = [time_col] + [cityid_col] + self.freq96
            # 生成统计列
            if isStat:
                df['MAX'] = df.loc[:,self.freq96].max(1)
                df['MIN'] = df.loc[:,self.freq96].min(1)
                df['AVG'] = df.loc[:,self.freq96].mean(1)
        elif len(df.columns) == 48+2:
            df.columns = [time_col] + [cityid_col] + self.freq48
            # 生成统计列
            if isStat:
                df['MAX'] = df.loc[:,self.freq48].max(1)
                df['MIN'] = df.loc[:,self.freq48].min(1)
                df['AVG'] = df.loc[:,self.freq48].mean(1)
        elif len(df.columns) == 24+2:
            df.columns = [time_col] + [cityid_col] + self.freq24
            # 生成统计列
            if isStat:
                df['MAX'] = df.loc[:,self.freq24].max(1)
                df['MIN'] = df.loc[:,self.freq24].min(1)
                df['AVG'] = df.loc[:,self.freq24].mean(1)
        
        # 将空值都转换为Nan
        df.fillna(np.nan,inplace=True)
            

        # 是否需要将Nan转换为null
        if isNa2Null:
            df.replace(np.nan,'null',inplace=True)
        
        # 筛选地市
        if city_id:
            df = df[df[cityid_col]==city_id]
        # 删除city_id列
        if isDelCitycol:
            df = df.drop(cityid_col,axis=1)
        
        return df
    

    def table2col(self,df:pd.DataFrame, time_col:str='DATE', y_col:str='load', freq:int=96,index_type:str='normal'):
        """
        此方法支持将97列、49列、25列日期+对应时间频次数据的Dataframe转换为1列索引为日期+指定时刻和对应时刻数据的Dataframe。
        同时，此方法支持将96时刻点转换为48时刻点

        Parameters:
        ----------
            df:Dataframe
                待转换的数据,数据列数必须为97、49、25列其中之一,同时,需包含日期列,日期格式为年月日。
            time_col:str
                用于定位时间列,默认为"DATE"
            y_col:str
                转换后信息列的列名
            freq:int
                需转换的时间频次,可填入96、48、24,默认为96
            index_type:str
                索引类型,如果为标准的日期格式则填'normal',如果为int格式,则填写'int'
        Returns:
        ----------
            DataFrame
        """

        # 如果freq的输入值不在96，48，24里面，则报错
        if freq not in [96,48,24]:
            raise ValueError('The "freq" can only be entered as 96, 48 or 24!')
        # 如果日期列不在df的列名中但与df的索引相同，则重置索引
        if time_col not in df.columns and time_col == df.index.name:
            df.reset_index(inplace=True)
        elif time_col not in df.columns and time_col != df.index.name:
            raise KeyError('"{}" is not in the column or index of "df" !'.format(time_col))
        # 判断日期是否存在重复，重复则保留第一个值，删除并打印重复日期索引，报警告
        
        # 输入的dataframe列数必须为97，49或25其中之一
        if len(df.columns) not in [97,49,25]:
            raise TypeError('The number of "df" columns needs to be 97, 49 or 25, please adjust the input dataframe')

        # 如果df时刻值列的数量比freq小，则报错
        if len(df.columns)-1 <freq:
            raise ValueError('Conversion from short time series to long time series is not supported !(e.g. From 24 time or 48 time --> 96 time, 24 time --> 48 time)')


        if len(df.columns)-1 == freq:
        # 如果df时刻值的列数量与freq相等，对df不做其他操作        
            df = df.copy()
        elif len(df.columns)-1 >freq and freq == 48:
        # 如果df时刻值的列数量与freq不相等，freq为48，那么将时刻值列处理为48列
            df = pd.concat([df[time_col],df.iloc[:,1::2]],axis=1)
        elif len(df.columns)-1 >freq and freq == 24:
        # 如果df时刻值的列数量与freq不相等，freq为48，那么将时刻值列处理为48列
            df = pd.concat([df[time_col],df.iloc[:,1::4]],axis=1)



        if not isinstance(df.loc[:, time_col].dtype, pd.Timestamp):
            df.loc[:, time_col] = df.loc[:, time_col].apply(lambda x: pd.Timestamp(x))
        if index_type == "int":
            df.loc[:, time_col] = df.loc[:, time_col].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8])

        df = df.rename(columns={time_col: 'DATE'}).set_index('DATE')
        df = df.asfreq('d')
        min_time = df.index.min()
        max_time = df.index.max() + pd.Timedelta(days=1) - pd.Timedelta(minutes=self.__num2freq_minutes[freq])
        df_values = df.values.flatten()
        df_out = pd.DataFrame(data=df_values,columns=[y_col],index=pd.date_range(min_time, max_time, freq=self.__num2freq_minutes_str[freq]))
        df_out.index.name = time_col
        return df_out

    
    def col2table(self,df:pd.DataFrame,time_col='DATE',info_col='load',freq=96):
        """
        此方法支持将竖向日期+96时刻/48时刻/24时刻与对应时刻信息数据的Dataframe进行横向展开为日期+96个时刻列/48个时刻列/24个时刻列
        同时,此方法支持将竖向96时刻点转换为横向48时刻点/24时刻点、竖向48时刻点转换为横向24时刻点的操作

        Parameters:
        ----------
            df:Dataframe
                待转换的数据,数据列数必须为97、49、25列其中之一,同时,需包含日期列,日期格式为年月日。
            time_col:str
                用于定位时间列,默认为"DATE"
            info_col:str
                除日期、时刻外的信息列,默认为'load'
            freq:int
                需转换的时间频次,可填入96、48、24,默认为96
        Returns:
        ----------
            DataFrame
        """
        # 输入的dataframe列数必须为97，49或25其中之一
        if freq not in [96,48,24]:
            raise ValueError('The value of "freq" can only be 96, 48, 24.')

        # 如果info_col不在df中，则报错
        if info_col not in df.columns:
            raise KeyError('"{}" is not in the column or index of "df" !'.format(info_col))
        
        # 如果time_col不在df的列中同时也不为索引，则报错
        if time_col not in df.columns and time_col != df.index.name:
            raise KeyError('"{}" is not in the column or index of "df" !'.format(time_col))
            
        # 如果time_col为索引，则将其变为正常列
        if time_col not in df.columns and time_col == df.index.name:
            df.reset_index(inplace=True)
        # 如果时刻值长度小于freq，则报错

        df = df[[time_col, info_col]].set_index(time_col).copy()

        if df.resample('d')[info_col].apply(list).apply(len).max() <freq:
            raise ValueError('Conversion from short time series to long time series is not supported !(e.g. From 24 time or 48 time --> 96 time, 24 time --> 48 time)')

        # 生成起始时间和结束时间
        s_time = pd.Timestamp(str(df.index.min())[0:10] + ' 00:00:00')
        e_time = pd.Timestamp(str(df.index.max())[0:10] + ' 23:45:00')


        # 生成一个空DataFrame，标题设定为日期+指定频率时刻
        df_table = pd.DataFrame(columns=[time_col] + self.num2freq[freq])

        df_table.loc[:, time_col] = pd.date_range(s_time, e_time, freq='1d')
        df_table.loc[:, time_col] = df_table.loc[:, time_col].apply(lambda x: str(x)[0:10])
        df_table.set_index(time_col, inplace=True)

        # 生成时间列表
        df_time = pd.DataFrame(data=pd.date_range(s_time, e_time, freq=self.__num2freq_minutes_str[freq]),columns=[time_col])

        # 按照指定频率提取数据
        df = pd.merge(df_time,df,how='left',left_on=time_col,right_index=True).set_index(time_col)


        for day in pd.date_range(s_time, e_time, freq='1d'):
            day = str(day)[0:10]
            data = df.loc[day].values.flatten().tolist()
            df_table.loc[day, :] = data
        
        return df_table

        