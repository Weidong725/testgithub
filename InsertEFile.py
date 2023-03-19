import pandas as pd 
from datetime import datetime 

class InsertEFile(object):
    """用于生成批量测算中的raw.e文件, 可支持批量插入自定义数据, 插入数据需存储为字典形式,key为数据标签,value为DataFrame

    raw.e文件内容中包含以下两类信息:
    \t 1.默认信息(不需要经常更改的信息): 已在 InsertEFile类下的def __init__()下配置好, 如需变更或添加新信息可在 def __init__()下添加self.xxxx变量信息, 添加数据格式可以为字典格式或DataFrame格式。
        
        \t (1) 字典格式: 字典格式的信息中必须存在名为'label'的键, label键的值为raw.e文件中'< >'内的数据标签名称, 在def __init__()中配置好后, 需在def SaveBaseInfo2E()方法下添加f.write(self.convertDict(self.xxxx))即可
        \t (2) DataFrame格式: DataFrame格式的数据需先在def __init__()下创建self.xxx变量, 通过pd.Dateframe自行创建DataFrame, 也可以文件形式(文件放在generate_efile文件夹下即可)通过pd.read_csv或pd.read_excel读取, 将文件路径为“./timeseries_tools/传入文件名称.csv”。创建好变量后, 需在def SaveBaseInfo2E()方法下添加f.write(self.convertDataFrame(self.xxxx,’数据标签名称’))。
        \t (3) 节假日、调休日的更新: 节假日、调休日信息已预先放置在timeseries_tools文件夹下的节假日.csv、调休日.csv文件中, 如需更新, 更新文件内容即可。
        
    \t 2.自定义信息(经常需要变更的数据): 如批量测试中经常需要替换历史负荷数据、历史温度数据和温度统计数据等。可将多个数据存储为字典形式, 字典的键为数据标签名称(raw.e文件中<>内的内容), 值为待插入数据, 存储好后，传入InsertEFile()中的batch_insert_dic参数内。\n

    Parameters
    ----------
        s_date
            批量测算的起始日期
        e_date
            批量测算的结束日期
        path_file
            E文件的保存路径
        batch_insert_dict:
            需要批量插入的数据及其标签, 在传入前需存储为 key:数据标签(String), value:待插入数据(Dataframe)的字典形式, 该参数默认为None。

    """
    def __init__(self,s_date,e_date,path_file,batch_insert_dict=None):
        """raw.e文件的固定信息,可按需要调整

        Parameters
        ----------
        s_date
            批量测算的起始日期。
        e_date
            批量测算的结束日期。
        path_file
            E文件的保存路径。
        batch_insert_dict, optional
            需要批量插入的数据及其标签, 在传入前需存储为key:数据标签(String), value:待插入数据(Dataframe)的字典形式, 该参数默认为None, by default None
        """        
        
        self.path_file = path_file
        self.s_date = s_date
        self.e_date = e_date
        self.batch_insert_dict = batch_insert_dict
        self.col_name = ['Date']+["T"+ "{:02d}".format(m) + "{:02d}".format(h) for m in range(0,24) for h in range(0,60,15)]
        # 基本信息
        self.base_info = {
            'PropertyID':   ['ForecastBeginDay','ForecastEndDay','ForecastNumDay','NewPoint','IsMultiThread' ,'Algorithm','NumOfRound']
            ,'PropertyName':['测试起始日'      ,'测试结束日'     ,'预测天数'       ,'新息点数' ,'是否启用多线程' ,'算法'     ,'训练轮次']
            ,'Value':       [self.s_date      ,self.e_date     ,1               ,38         ,0              ,109       ,300]
            ,'label':'ControlParameterBatchTest'
        }
        # 参数搜索配置信息
        self.search_param = {
            'PropertyID': ['Time_Interval','Thresh_Day','Recent_Days','Moment_Use_Day','Day_Acc_Thresh']
            ,'ValueRange':['(4)'          ,'(1.5)'     ,'(400)'      ,'(60)'          ,'(80)']
            ,'Type':      ['None'         ,'None'      ,'None'       ,'None'          ,'None']
            ,'label':'SearchParameter'
        }
        # 参数搜索
        self.opt_param={
            'PropertyID':   ['IsSearchParam'  ,'Max_Search_Num'  ,'ThreadNum']
            ,'PropertyName':['是否启用参数搜索' ,'最大搜索次数'     ,'线程数']
            ,'Value':       [1                ,1                 ,3]
            ,'label':'OptimizationParameter'
        }
        # ---------如果使用的为神经网络模型，以下数据即使入模不需要也需要有标签和数据标题-------------
        # 节假日、调休日信息
        self.holidays = pd.read_csv('./timeseries_tools/节假日信息.csv')
        self.agjustdays = pd.read_csv('./timeseries_tools/调休日信息.csv')
        self.datenote = pd.DataFrame({'Date':['20210101'],'Cause':['疫情']})
        # 算法
        self.algo109 = pd.DataFrame({},columns=self.col_name)
        # 湿度
        self.humidity = pd.DataFrame({},columns=self.col_name)
        # 风速
        self.wind = pd.DataFrame({},columns=self.col_name)
        # 冰雹
        self.precipitation = pd.DataFrame({},columns=self.col_name)
        # 湿度统计
        self.humidityStat = pd.DataFrame({},columns=['Date','AVG'])
        # 风速统计
        self.windStat = pd.DataFrame({},columns=['Date','AVG'])
        # 冰雹统计数据
        self.precipitationStat = pd.DataFrame({},columns=['Date','AVG'])
        
    def __convertDict(self,temp_dict:dict):
        """将固定信息中dict形式的数据转为指定格式

        Parameters
        ----------
        temp_dict
            数据字典, 必须包含名为'label'的键
        """

        if not isinstance(temp_dict,dict):
            raise TypeError("需要传入的数据格式为dict")
        temp_df = pd.DataFrame(temp_dict)
        temp_df.insert(0,'@',"#")
        label_start = '\n'+'<'+temp_dict['label']+'>'+'\n'
        label_end = '\n'+'<'+'/'+temp_dict['label']+'>'+'\n'
        header = '  '.join(temp_df.drop('label',axis=1).columns) + '\n'
        content = temp_df.loc[:,temp_df.columns!='label'].to_string(header=False,index=False)
        result_content = label_start + header + content + label_end
        return result_content
    
    def __convertDataFrame(self,temp_df:pd.DataFrame,label_str:str):
        """将传入的dataframe类型数据转换为.e文件所需要的形式

        Parameters
        ----------
        temp_df
            待插入raw文件的dataframe
        label_str
            待插入数据的标签
        """        

        if not isinstance(temp_df,pd.DataFrame):
            raise TypeError("需要传入的数据格式为DataFrame")
            
        temp_df.insert(0,'@',"#")
        label_start = '\n'+'<'+ label_str +'>'+'\n'
        label_end = '\n'+'<'+'/'+label_str +'>'+'\n'
        header = '  '.join(temp_df.columns) + '\n'
        content = temp_df.to_string(header=False,index=False)
        if len(temp_df) <=0:
            result_content = label_start + header +label_end
        else:
            result_content = label_start + header + content + label_end
        return result_content

    def __SaveBaseInfo2E(self):
        """将默认信息存储至.e文件中
        """        
        with open(self.path_file,'w',encoding='utf8') as f:
            f.write('<! Grid=调度口径 Type=短期正常日负荷预测 Time= {}!> \n'.format(datetime.now().replace(microsecond=0)))

            f.write(self.__convertDict(self.base_info))
            f.write(self.__convertDict(self.search_param))
            f.write(self.__convertDict(self.opt_param))
            f.write(self.__convertDataFrame(self.holidays,'HolidayInfo'))
            f.write(self.__convertDataFrame(self.agjustdays,'AdjustedWorkday'))
            f.write(self.__convertDataFrame(self.datenote,'DateNotIncluded'))
            f.write(self.__convertDataFrame(self.algo109,'Algo109'))
            f.write(self.__convertDataFrame(self.humidity,'Humidity'))
            f.write(self.__convertDataFrame(self.wind,'Wind'))
            f.write(self.__convertDataFrame(self.precipitation,'Precipitation'))
            f.write(self.__convertDataFrame(self.humidityStat,'HumidityStat'))
            f.write(self.__convertDataFrame(self.windStat,'WindStat'))
            f.write(self.__convertDataFrame(self.precipitationStat,'PrecipitationStat'))

    def __CustomInsert(self):
        """将自定义信息批量插入.e文件中
        """        

        if not self.batch_insert_dict:
            print('---- 无自定义批量插入数据')
        else:
            if not isinstance(self.batch_insert_dict,dict):
                raise TypeError("batch_insert_dict传入值有误，请传入dict类型")
            with open(self.path_file,'a',encoding='utf8') as f:
                for label,df in self.batch_insert_dict.items():
                    if not isinstance(df,pd.DataFrame):
                        raise TypeError("batch_insert_dict 的value格式有误,需要传入Dataframe类型")
                    elif not isinstance(label,str):
                        raise TypeError("batch_insert_dict 的key格式有误,需要传入str类型")
                    else:
                        f.write(self.convertDataFrame(df,label))
                        print('{}信息插入成功'.format(label))
                print('---- 自定义信息写入成功 ')
    
    def GenerateEfile(self):
        """生成最终的.e文件"""
        
        self.__SaveBaseInfo2E()
        print('---- 基本信息写入成功 ')
        self.__CustomInsert()