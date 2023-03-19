import matplotlib.pyplot as plt
from timeseries_tools.TimeSeriesTransform import TimeSeriesTransform as tst
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class TimeSeriseTestReport(object):
    """
    用于输出批量测算各类统计值和图示
    
    """    
    def __init__(self,date_col:str):
                            
        """初始化96时刻列名称,节假日日期信息及定位日期列

        Parameters
        ----------
        date_col: 日期列名称
        """        
        self.date_col = date_col
        self.freq96 = ["T"+ "{:02d}".format(m) + "{:02d}".format(h) for m in range(0,24) for h in range(0,60,15)]
        self.holidays = pd.read_csv(r'./timeseries_tools/节假日信息.csv'
                                    ,usecols=['Date'],converters={'Date':lambda x:str(pd.to_datetime(x))[0:10]},squeeze=True)

    def __trans2merge(self,real_load:pd.DataFrame,fc_load:pd.DataFrame):        
        """用于将日期+96时刻的实际负荷与预测负荷转为2列竖向数据, 再按照日期索引进行合并

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式

        Returns
        -------
            形如日期(self.date_col), 实际负荷值(LOAD_real), 预测负荷值的Dataframe(LOAD_pre)
        """        
        real_load = tst().table2col(df=real_load,time_col=self.date_col,y_col='LOAD')
        fc_load = tst().table2col(df=fc_load,time_col=self.date_col,y_col='LOAD')
        merge_result = pd.merge(real_load,fc_load,how='inner',left_index=True,right_index=True,suffixes=['_real','_pre'])
        return merge_result

    def RMSPE(self,real_load, fc_load):
        """用于计算模型精度,精度计算方式为1-RMSPE

        Parameters
        ----------
        real_load: 实际负荷
        fc_load: 预测负荷
        """                
        fc_load = np.array(fc_load)
        real_load = np.array(real_load)
        n = len(real_load)
        temp = np.square((fc_load - real_load)/real_load).sum()
        score = np.sqrt(temp/n)
        return 1-score

    def TimeShareEval(self,real_load:pd.DataFrame,fc_load:pd.DataFrame,isDelHoliday=True)->pd.DataFrame:      
        """用于计算96时刻每时刻平均rmspe, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        isDelHoliday, optional
            是否剔除节假日, by default True
        """        
        metric_dict = {}
        col_name = [self.date_col]+self.freq96
        # 重置列名
        real_load.columns = col_name
        fc_load.columns = col_name

        if isDelHoliday:# 剔除节假日日期
            real_load = real_load.loc[~real_load[self.date_col].isin(self.holidays)]
            fc_load = fc_load.loc[~fc_load[self.date_col].isin(self.holidays)]

        for point in self.freq96:
            r = real_load[['DATE',point]].dropna()
            p = fc_load[['DATE',point]].dropna()
            rp = pd.merge(r,p,how='inner',left_on=self.date_col,right_on=self.date_col)
            rmspe = self.RMSPE(rp.set_index(self.date_col).iloc[:,0],rp.set_index(self.date_col).iloc[:,1])
            metric_dict[point] = np.mean(rmspe)
        result =  pd.DataFrame.from_dict(metric_dict,orient='index',columns=['rmspe_mean'])

        print('==== 各时刻平均精度 ====')
        print(result)

        return result

    
    def WetherHolidayAcc(self,real_load:pd.DataFrame,fc_load:pd.DataFrame):
        """用于计算节假日的模型平均精度和非节假日的模型平均精度, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        """        
        # 计算节假日精度
        r_h = real_load.loc[real_load[self.date_col].isin(self.holidays)]
        p_h = fc_load.loc[fc_load[self.date_col].isin(self.holidays)]
        # 转换并合并
        rp_h = self.__trans2merge(r_h,p_h)

        rmspe_h = rp_h.resample('d').apply(lambda x:self.RMSPE(x['LOAD_real'],x['LOAD_pre'])).to_frame(name='rmspe')

        # 计算非节假日精度
        r_n = real_load.loc[~real_load[self.date_col].isin(self.holidays)]
        p_n = fc_load.loc[~fc_load[self.date_col].isin(self.holidays)]
        # 转换并合并
        rp_n = self.__trans2merge(r_n,p_n)

        rmspe_n = rp_n.resample('d').apply(lambda x:self.RMSPE(x['LOAD_real'],x['LOAD_pre'])).to_frame(name='rmspe')

        print('节假日平均精度：{}，非节假日平均精度：{}'.format(rmspe_h['rmspe'].mean(),rmspe_n['rmspe'].mean())) 

        return float(rmspe_h.mean()),float(rmspe_n.mean())


    def MonthlyAcc(self,real_load:pd.DataFrame,fc_load:pd.DataFrame,isDelHoliday=True):
        """用于计算模型每月平均精度, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        isDelHoliday, optional
            是否剔除节假日, by default True
        """        
        if isDelHoliday:# 剔除节假日日期
            real_load = real_load.loc[~real_load[self.date_col].isin(self.holidays)]
            fc_load = fc_load.loc[~fc_load[self.date_col].isin(self.holidays)]

        result = self.__trans2merge(real_load,fc_load)
        result = pd.merge(real_load,fc_load,how='inner',left_index=True,right_index=True,suffixes=['_real','_pre'])
        rmspe_date = result.resample('d').apply(lambda x:self.RMSPE(x['LOAD_real'],x['LOAD_pre'])).to_frame(name='rmspe')
        rmspe_month = rmspe_date.resample('M')['rmspe'].mean().to_frame().reset_index()
        rmspe_month[self.date_col] = rmspe_month[self.date_col].dt.strftime('%Y-%m')
        print('==== 每月平均精度 ==== ')
        print(rmspe_month)
        return rmspe_month
    
    def PeakValleyAcc(self,real_load:pd.DataFrame,fc_load:pd.DataFrame,time_interval=[[1,7],[8,12],[13,16],[17,19],[20,23]],isDelHoliday=True):
        """用于计算不同时间段最大值与最小值的平均精度, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        time_interval, optional
            时间段列表, 可采用列表嵌套的方式输入多个时间段 by default [[1,7],[8,12],[13,16],[17,19],[20,23]]
        isDelHoliday, optional
            是否剔除节假日, by default True
        """        

        if isDelHoliday:# 剔除节假日日期
            real_load = real_load.loc[~real_load[self.date_col].isin(self.holidays)]
            fc_load = fc_load.loc[~fc_load[self.date_col].isin(self.holidays)]

        rf_merge = self.__trans2merge(real_load,fc_load)

        rf_merge['Month'] = rf_merge[self.date_col].dt.month #提取月份 
        rf_merge['Day'] = rf_merge[self.date_col].dt.day #提取天数 
        rf_merge['Hour'] = rf_merge[self.date_col].dt.hour #提取小时
        print('==== 高峰低谷时间段平均精度 ====')
        max_val,min_val =[],[]
        for times in time_interval:
            # 提取指定时间区间
            time_filtered = rf_merge.loc[(rf_merge['Hour'] >=times[0]) & (rf_merge['Hour'] <times[1]),:]
            # 提取最大值索引
            max_value = pd.DataFrame(time_filtered.groupby(['Month','Day'])['LOAD_real'].max()).reset_index()
            min_value = pd.DataFrame(time_filtered.groupby(['Month','Day'])['LOAD_real'].min()).reset_index()

            result_max = pd.merge(rf_merge,max_value,on=['Month','Day','LOAD_real'],how='inner')[[self.date_col,'LOAD_real','LOAD_pre']].dropna()
            result_min = pd.merge(rf_merge,min_value,on=['Month','Day','LOAD_real'],how='inner')[[self.date_col,'LOAD_real','LOAD_pre']].dropna()

            rmspe_max = self.RMSPE(result_max['LOAD_real'],result_max['LOAD_pre'])
            rmspe_min = self.RMSPE(result_min['LOAD_real'],result_min['LOAD_pre'])
            max_val.append([times,rmspe_max])
            min_val.append([times,rmspe_min])
            print('{}点至{}点最大值平均精度:{:.5f}'.format(times[0],times[1],rmspe_max))
            print('{}点至{}点最小值平均精度:{:.5f}'.format(times[0],times[1],rmspe_min))
        return max_val,min_val
    
    def WeeklyAcc(self,real_load:pd.DataFrame,fc_load:pd.DataFrame,isDelHoliday=True):
        """用于统计不同星期类型(工作日、休息日)的平均精度, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        isDelHoliday, optional
            是否剔除节假日, by default True
        """        
        if isDelHoliday:# 剔除节假日日期
            real_load = real_load.loc[~real_load[self.date_col].isin(self.holidays)]
            fc_load = fc_load.loc[~fc_load[self.date_col].isin(self.holidays)]

        rf_merge = self.__trans2merge(real_load,fc_load).reset_index()
        print('==== 各星期类型平均精度 ====')
        val = []
        for week in range(0,7):
            temp_week = rf_merge[rf_merge[self.date_col].dt.weekday==week].set_index(self.date_col).resample('d').apply(lambda x:self.RMSPE(x['LOAD_real'],x['LOAD_pre']))
            rmspe = temp_week.mean()
            print('周{}:'.format('日' if week+1==7 else week+1),rmspe)
            val.append([week+1,rmspe])
        return val
        
    
    def plot1Picture(self,real_load:pd.DataFrame,fc_load:pd.DataFrame,real_weather:pd.DataFrame,fc_weather:pd.DataFrame,isSave=False):
        """将实际负荷,预测负荷,实际气象,预测气象绘制在双轴折线图上, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        real_weather
            实际气象, 数据格式需为日期+96时刻气象值的形式
        fc_weather
            预测气象, 数据格式需为日期+96时刻气象值的形式
        isSave, optional
            是否保存为可交互html文件,如为True,则会在当前目录下生成名为plot1Picture.html的文件, by default False
        """        
        import plotly
        import plotly.graph_objects as go

        real_load = tst().table2col(df=real_load,time_col=self.date_col,y_col='LOAD')
        fc_load = tst().table2col(df=fc_load,time_col=self.date_col,y_col='LOAD')
        real_weather = tst().table2col(df=real_weather,time_col=self.date_col,y_col='TEMP')
        fc_weather = tst().table2col(df=fc_weather,time_col=self.date_col,y_col='TEMP')

        real_load = go.Scatter(
            x=real_load.index, y=real_load.iloc[:,0], mode='lines'
            , name='实际负荷',line=dict(dash='solid')
            ,opacity=0.9,yaxis='y1'
            )
        fc_load = go.Scatter(
            x=fc_load.index, y=fc_load.iloc[:,0], mode='lines'
            , name='预测负荷',line=dict(dash='longdashdot')
            ,opacity=0.9,yaxis='y1'
            )
        real_weather = go.Scatter(
            x=real_weather.index, y=real_weather.iloc[:,0], mode='lines'
            , name='实际气象',line=dict(dash='solid')
            ,opacity=0.9,yaxis='y2'
            )
        fc_weather = go.Scatter(
            x=fc_weather.index, y=fc_weather.iloc[:,0], mode='lines'
            , name='预测气象',line=dict(dash='longdashdot')
            ,opacity=0.9,yaxis='y2'
            )
        data = [real_load,fc_load,real_weather,fc_weather]

        layout = go.Layout(title="负荷温度曲线",
                    yaxis=dict(title="负荷值"),
                    yaxis2=dict(title="温度值", overlaying='y', side="right"),
                    legend=dict(x=0, y=1, font=dict(size=10, color="black")))
        fig = go.Figure(data=data, layout=layout)
        fig.show()

        if isSave:
            plotly.offline.plot(fig, filename='./plot1Picture.html')
        
    def plot2Picture(self,real_load:pd.DataFrame,fc_load:pd.DataFrame,real_weather:pd.DataFrame,fc_weather:pd.DataFrame,isSave=True):
        """
        用于将实际负荷, 预测负荷绘制在子图1。将实际气象, 预测气象绘制在子图2上。数据格式需为日期+96时刻负荷值的形式。

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        real_weather
            实际气象, 数据格式需为日期+96时刻气象值的形式
        fc_weather
            预测气象, 数据格式需为日期+96时刻气象值的形式
        isSave, optional
            是否保存为可交互html文件,如为True,则会在当前目录下生成名为plot2Picture.html的文件, by default True
        """        
        import plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        real_load = tst().table2col(df=real_load,time_col=self.date_col,y_col='LOAD')
        fc_load = tst().table2col(df=fc_load,time_col=self.date_col,y_col='LOAD')
        real_weather = tst().table2col(df=real_weather,time_col=self.date_col,y_col='TEMP')
        fc_weather = tst().table2col(df=fc_weather,time_col=self.date_col,y_col='TEMP')

        fig = make_subplots(rows=2,cols=1,subplot_titles=["实际负荷、预测负荷曲线", "实际温度、预测温度曲线"],shared_xaxes=True)
        opacity=0.9
        fig.add_trace(
            go.Scatter(x=real_load.index, y=real_load.iloc[:,0],mode='lines', name='实际负荷',opacity=opacity)
            ,row=1,col=1
        )
        fig.add_trace(
            go.Scatter(x=fc_load.index, y=fc_load.iloc[:,0],mode='lines', name='预测负荷',opacity=opacity)
            ,row=1,col=1
        )
        fig.add_trace(
            go.Scatter(x=real_weather.index, y=real_weather.iloc[:,0],mode='lines', name='实际温度',opacity=opacity)
            ,row=2,col=1
        )
        fig.add_trace(
            go.Scatter(x=fc_weather.index, y=fc_weather.iloc[:,0],mode='lines', name='预测温度',opacity=opacity)
            ,row=2,col=1
        )

        fig.show()
        if isSave:
            plotly.offline.plot(fig, filename='./plot2Picture.html')

    def contrastAlgo1Plot(self,real_load:pd.DataFrame,newalgo_load:pd.DataFrame,oldalgo_load:pd.DataFrame,isSave=False):
        """用于绘制新旧算法预测负荷结果与实际负荷的对比图像, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        newalgo_load
            新算法预测负荷结果, 数据格式需为日期+96时刻负荷值的形式
        oldalgo_load
            旧算法预测负荷结果, 数据格式需为日期+96时刻负荷值的形式
        isSave, optional
            是否保存为可交互html文件,如为True,则会在当前目录下生成名为contrastAlgo1Plot.html的文件, by default False
        """        
        import plotly
        import plotly.graph_objects as go

        real_load = tst().table2col(df=real_load,time_col=self.date_col,y_col='LOAD')
        newalgo_load = tst().table2col(df=newalgo_load,time_col=self.date_col,y_col='LOAD')
        oldalgo_load = tst().table2col(df=oldalgo_load,time_col=self.date_col,y_col='LOAD')


        real_load = go.Scatter(
            x=real_load.index, y=real_load.iloc[:,0], mode='lines'
            , name='实际负荷',line=dict(dash='solid')
            ,opacity=0.9
            )
        newalgo_load = go.Scatter(
            x=newalgo_load.index, y=newalgo_load.iloc[:,0], mode='lines'
            , name='新算法预测负荷',line=dict(dash='longdashdot')
            ,opacity=0.9
            )
        oldalgo_load = go.Scatter(
            x=oldalgo_load.index, y=oldalgo_load.iloc[:,0], mode='lines'
            , name='旧算法预测负荷',line=dict(dash='solid')
            ,opacity=0.9
            )

        data = [real_load,newalgo_load,oldalgo_load]

        layout = go.Layout(title="新旧算法对比曲线",
                    yaxis=dict(title="负荷值"),
                    legend=dict(x=0, y=1, font=dict(size=10, color="black")))
        fig = go.Figure(data=data, layout=layout)
        fig.show()

        if isSave:
            plotly.offline.plot(fig, filename='./contrastAlgo1Plot.html')

    def contrastAlgo2Plot(self,real_load:pd.DataFrame,newalgo_load:pd.DataFrame,oldalgo_load:pd.DataFrame,real_weather:pd.DataFrame,fc_weather:pd.DataFrame,isSave=True):
        """用于将新旧算法预测负荷结果与实际负荷的对比图像绘制在子图1上, 将实际气象与预测气象绘制在子图2上, 数据格式需为日期+96时刻负荷值的形式

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        newalgo_load
            新算法预测负荷, 数据格式需为日期+96时刻负荷值的形式
        oldalgo_load
            旧算法预测负荷, 数据格式需为日期+96时刻负荷值的形式
        real_weather
            实际气象, 数据格式需为日期+96时刻气象值的形式
        fc_weather
            预测气象, 数据格式需为日期+96时刻气象值的形式
        isSave, optional
            是否保存为可交互html文件,如为True,则会在当前目录下生成名为contrastAlgo2Plot.html的文件, by default True
        """        
        import plotly
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        real_load = tst().table2col(df=real_load,time_col=self.date_col,y_col='LOAD')
        newalgo_load = tst().table2col(df=newalgo_load,time_col=self.date_col,y_col='LOAD')
        oldalgo_load = tst().table2col(df=oldalgo_load,time_col=self.date_col,y_col='LOAD')

        real_weather = tst().table2col(df=real_weather,time_col=self.date_col,y_col='TEMP')
        fc_weather = tst().table2col(df=fc_weather,time_col=self.date_col,y_col='TEMP')

        fig = make_subplots(rows=2,cols=1,subplot_titles=["实际负荷与新旧算法预测负荷对比曲线", "实际温度、预测温度曲线"],shared_xaxes=True)
        opacity=0.9

        fig.add_trace(
            go.Scatter(x=real_load.index, y=real_load.iloc[:,0],mode='lines', name='实际负荷',opacity=opacity)
            ,row=1,col=1
        )
        fig.add_trace(
            go.Scatter(x=newalgo_load.index, y=newalgo_load.iloc[:,0],mode='lines', name='新算法预测负荷',opacity=opacity)
            ,row=1,col=1
        )
        fig.add_trace(
            go.Scatter(x=oldalgo_load.index, y=oldalgo_load.iloc[:,0],mode='lines', name='旧算法预测负荷',opacity=opacity)
            ,row=1,col=1
        )
        fig.add_trace(
            go.Scatter(x=real_weather.index, y=real_weather.iloc[:,0],mode='lines', name='实际温度',opacity=opacity)
            ,row=2,col=1
        )
        fig.add_trace(
            go.Scatter(x=fc_weather.index, y=fc_weather.iloc[:,0],mode='lines', name='预测温度',opacity=opacity)
            ,row=2,col=1
        )

        fig.show()
        if isSave:
            plotly.offline.plot(fig, filename='./contrastAlgo2Plot.html')

    def outputReport(self,real_load,fc_load,time_interval=[[1,7],[8,12],[13,16],[17,19],[20,23]],path='./TestReport.txt',isDelHoliday=True):
        """输出全部测算信息

        Parameters
        ----------
        real_load
            实际负荷, 数据格式需为日期+96时刻负荷值的形式
        fc_load
            预测负荷, 数据格式需为日期+96时刻负荷值的形式
        time_interval, optional
            时间区间, 用于计算不同时间区间的最大负荷与最小负荷的平均精度, 可采用列表嵌套的方式输入多个时间段 by default [[1,7],[8,12],[13,16],[17,19],[20,23]]
        path, optional
            输出文件存储路径, by default './TestReport.txt'
        isDelHoliday, optional
            是否剔除节假日, by default True
        """        
        holiday_acc,no_holiday_acc = self.WetherHolidayAcc(real_load,fc_load)
        every_points_acc = self.TimeShareEval(real_load,fc_load,isDelHoliday)
        every_month_acc = self.MonthlyAcc(real_load,fc_load,isDelHoliday)
        time_interval_max,time_interval_min = self.PeakValleyAcc(real_load,fc_load,time_interval,isDelHoliday)
        week_day_acc = self.WeeklyAcc(real_load,fc_load,isDelHoliday)

        with open(path,'w+') as f:
            print('节假日平均精度{},剔除节假日平均精度{}'.format(holiday_acc,no_holiday_acc),file=f)
            print('==== 分时刻平均精度 ==== \n',every_points_acc.to_string(),file=f)
            print('==== 每月平均精度 ==== \n',every_month_acc.to_string(index=False),file=f)

            print('==== 分时段最大负荷平均精度 ====',file=f)
            for times,acc in time_interval_max:
                print('{}点至{}点最大负荷平均精度：{}'.format(times[0],times[1],acc),file=f)

            print('==== 分时段最大低荷平均精度 ====',file=f)
            for times,acc in time_interval_min:
                print('{}点至{}点最低负荷平均精度：{}'.format(times[0],times[1],acc),file=f)
            print('==== 各星期类型平均精度 ====',file=f)
            for w,acc in week_day_acc:
                print('星期{}：平均精度{}'.format(w,acc),file=f)
