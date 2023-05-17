# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
import pandas
import pandas as pd
import psycopg2
import psycopg2.extras
from pykrige.ok import OrdinaryKriging
from matplotlib import pyplot as plt
import gstools as gs

from src.utilities.logger import Logger

# 取消numpy数字结果 以科学计数法显示
np.set_printoptions(suppress=True)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
out_path = root_path + '\\test\\temp'
# logger文件路径
log_all = out_path + '\\calc_rain_logging.log'
# 格网点数量
grid_number = 200

# 很重要，地理分析工具，参数都可调整
model = gs.Gaussian(
    dim=2, var=1, nugget=0.1, len_scale=0.05
)

# 连接数据库
def connect_to_database(database_info):
    # 连接postgre数据库
    try:
        postgresdb = psycopg2.connect(database=database_info["dbname"],
                                      host=database_info["host"],
                                      user=database_info["user"],
                                      password=database_info["password"],
                                      port=database_info["port"])
        Logger(log_all, level='info').logger.info('connect success!')
        return {"status": True, "data": postgresdb}
    except Exception as e:
        Logger(log_all, level='error').logger.error(e)
        return {"status": False, "data": e}

# 克里金插值
def rain_kriging(lats, lons, data):
    # 生成经纬度网格点
    rain_value = ''
    grid_lon = np.linspace(118.0, 120.5, grid_number)
    grid_lat = np.linspace(25.0, 27.0, grid_number)
    # grid_lon = np.linspace(min(lons), max(lons), grid_number)
    # grid_lat = np.linspace(min(lats), max(lats), grid_number)
    # 克里金插值
    try:
        OK = OrdinaryKriging(
            lons,
            lats,
            data,
            variogram_model=model,
            verbose=False,
            enable_plotting=False,
            coordinates_type="geographic"
        )
        # OK = OrdinaryKriging(lons, lats, data, variogram_model='gaussian', nlags=6)
        z1, ss1 = OK.execute('grid', grid_lon, grid_lat)

        OK = OrdinaryKriging(
            lons, lats, data, variogram_model=model, verbose=False,
            enable_plotting=False, nlags=3
        )
        # Execute on grid:
        z2, ss2 = OK.execute("grid", grid_lon, grid_lat)
        cf = plt.contourf(grid_lon, grid_lat, z1, cmap=plt.cm.Blues)
        plt.colorbar(cf)
        plt.show()
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(z1, extent=[118.0, 120.5, 25.0, 27.0], origin="lower")
        # ax1.set_title("geo-coordinates")
        # ax2.imshow(z2, extent=[118.0, 120.5, 25.0, 27.0], origin="lower")
        # ax2.set_title("non geo-coordinates")
        # plt.show()

        xgrid, ygrid = np.meshgrid(grid_lon, grid_lat)
        df_grid = pd.DataFrame(dict(lon=xgrid.flatten(), lat=ygrid.flatten()))
        df_grid["Krig_gaussian"] = z1.flatten()
        # 格式化插值结果
        point_list = []
        print(df_grid.values)
        for grid in df_grid.values:
            point = '(' + str(grid[0]) + ',' + str(grid[1]) + ',' + str(grid[2]) + ')'
            point_list.append(point)
        # 将插值点字符串用英文';'连接
        rain_value = ';'.join(point_list)
    except Exception as e:
        if max(data) == 0 and min(data) == 0:
            point_list = []
            for (lon, lat) in zip(grid_lon, grid_lat):
                point = '(' + str(lat) + ',' + str(lon) + ',' + str(0) + ')'
                point_list.append(point)
            rain_value = ';'.join(point_list)
        else:
            rain_value = False
            Logger(log_all, level='error').logger.error(e)
    finally:
        # 返回字符串格式的插值结果：(lat,lon,value);(lat,lon,value)
        return rain_value

# 数据库读取数据+写入数据
def get_rain_data(start_time, end_time, postgresdb=None):
    try:
        # 查询
        # query_sql = "SELECT sum(rain_sum_10m_value) as rainvalue, stationlat, stationlon, stationcode " \
        #             "FROM t_sdwt_wo_his where rain_sum_10m_begintime " \
        #             "BETWEEN '{0}' AND '{1}' and rain_sum_10m_endtime BETWEEN '{0}' AND '{1}' " \
        #             "and rain_sum_10m_value is not null and rain_sum_10m_value != 0 and stationlat is not null " \
        #             "GROUP BY stationlat, stationlon, stationcode".format(start_time, end_time)
        query_sql = "SELECT sum(rain_sum_10m_value) as rainvalue, stationlat, stationlon, stationcode " \
                    "FROM t_sdwt_wo_his where rain_sum_10m_begintime " \
                    "BETWEEN '{0}' AND '{1}' " \
                    "and rain_sum_10m_value is not null and stationlat is not null " \
                    "GROUP BY stationlat, stationlon, stationcode".format(start_time, end_time)
        print(query_sql)
        # 从雨水表表中获取数据
        query_cursor = postgresdb.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query_cursor.execute(query_sql)
        rain_rows = query_cursor.fetchall()
        query_cursor.close()
        if len(rain_rows) <= 0:
            Logger(log_all, level='info').logger.info('find no data!')
            return False
        # 解析结果
        lats = []
        lons = []
        rain_values = []
        for rain in rain_rows:
            print(float(rain['stationlat']), float(rain['stationlon']), round(float(rain['rainvalue']), 3))
            lats.append(float(rain['stationlat']))
            lons.append(float(rain['stationlon']))
            rain_values.append(round(float(rain['rainvalue']), 2))
        # kriging插值，需要将一般array格式转换成pandas.Series格式
        rain_value_kriging = rain_kriging(pandas.Series(lats),
                                          pandas.Series(lons),
                                          pandas.Series(rain_values))
        # 如果插值失败不写入数据库
        if rain_value_kriging is False:
            return False
        # 将数据写入数据库表
        insert_cursor = postgresdb.cursor()
        insert_sql = "insert into stat_sdwt_wo (start_time,end_time,rain_value) " \
                        "values('{0}', '{1}', '{2}')".format(start_time, end_time, rain_value_kriging)
        # insert_cursor.execute(insert_sql)
        # 关闭游标
        insert_cursor.close()
        # 提交插入
        postgresdb.commit()
        # 关闭数据库连接
        postgresdb.close()
        return True

    except Exception as e:
        Logger(log_all, level='error').logger.error(e)
        print(e)


# flask服务
if __name__ == '__main__':
    # print(sys.argv[1], sys.argv[2])
    # start_time = sys.argv[1]
    # end_time = sys.argv[2]
    end_time = "2022-05-24 09:00:00"
    start_time = "2022-05-24 06:00:00"

    # 连接数据库 （外网）
    database_info = {
        'dbname': '', #数据库名称
        'host': '', #数据库ip
        'user': '', #数据库用户名
        'password': '', #数据连接密码
        'port': '', #数据库端口号
    }
    db_result = connect_to_database(database_info)
    if db_result['status'] is False:
        print({'status': 'false1', 'data': str(db_result['data'])})
    else:
        postgresdb = db_result['data']
        # 写sql获取雨量表数据
        result = get_rain_data(start_time, end_time, postgresdb)
        print(result)
