import pandas as pd
from prophet import Prophet
import logging
import cmdstanpy




#avoiding warnings from pandas
import warnings
warnings.filterwarnings('ignore')
logging.getLogger("prophet").setLevel(logging.CRITICAL)
'''
encapsulated model component
'''
ESKDALEMUI_data_renew = pd.read_csv('ESKDALEMUI_data_renew.csv')
ESKDALEMUI_data_renew['DATE'] = ESKDALEMUI_data_renew['DATE'].astype(int)
ESKDALEMUI_data_renew = ESKDALEMUI_data_renew[ESKDALEMUI_data_renew['DATE']<20220418]
ESKDALEMUI_data_renew['DATE'] = ESKDALEMUI_data_renew['DATE'].astype(str)
ESKDALEMUI_data_renew = ESKDALEMUI_data_renew.dropna()
model_data = ESKDALEMUI_data_renew

LONDON_data_renew = pd.read_csv('LONDON_data_renew.csv')
LONDON_data_renew['DATE'] = LONDON_data_renew['DATE'].astype(int)
LONDON_data_renew = LONDON_data_renew[LONDON_data_renew['DATE']<20200418]
LONDON_data_renew['DATE'] = LONDON_data_renew['DATE'].astype(str)
model_data_BARINDISI = LONDON_data_renew




#  Train the model Return data for a specific date
def train_model(data_train,data_test):
    # Set the log level to WARNING
    # prophet Suppress log output
    logging.getLogger(cmdstanpy.__name__).setLevel(logging.CRITICAL)

    data_train['ds'] = pd.to_datetime(data_train['ds'], format='%Y%m%d')  # Time conversion
    meanD = data_train['y'].mean()
    stdD = data_train['y'].std()
    data_train['y'] = (data_train['y'] - meanD ) / (stdD)  # standardization
    m = Prophet()
    # Add multiple variables
    m.add_regressor('CC')
    m.add_regressor('QQ')
    m.add_regressor('SS')
    print('-----------------------------------------------------------------------------------')
    m.fit(data_train)
    print('-----------------------------------------------------------------------------------')
    forecast = m.predict(data_test.drop(columns='y'))
    forecast['yhat'] = forecast['yhat']*stdD + meanD #recover data

    return forecast


def deal_data(model_data_input,input3):
    # 划分数据级
    model_data_test = model_data_input[['DATE',input3,'CC','QQ','SS']][-7300:]
    model_data_test['DATE'] = pd.to_datetime(model_data_test['DATE'], format='%Y%m%d') #time conversion
    model_data_test = model_data_test.rename(columns={'DATE':'ds', input3:'y'})

    model_data_train = model_data_input[['DATE',input3,'CC','QQ','SS']][:-7300]
    model_data_train = model_data_test.rename(columns={'DATE':'ds',input3:'y'})
    model_data_train['ds'] = pd.to_datetime(model_data_train['ds'], format='%Y%m%d') #time conbersion
    return model_data_train,model_data_test

#  Filtering the data because there is no data on March 19, 2022; it's all empty
def get_endData(forecast,input2):
    try:
        data = float(forecast[forecast['ds'] == input2]['yhat'])/10
    except:
        data = float(forecast[forecast['ds'] == '2022-3-18']['yhat'])/10
    return data

def model_test(input1,input2,input3):#input1-city input2-date input3- level
    
    answer = ''
    if('edinburgh' in input1):
        if('high' in  input3):
            model_data_train,model_data_test = deal_data(model_data,'TX')
            forecast = train_model(model_data_train,model_data_test)
            end_data = get_endData(forecast,input2)
            # print(end_data)
 
            answer ='In {} on {}, {} will be {:.2f}.'.format(input1,str(input2).replace("2022","2023"),"the highest temperature",end_data)

        elif('low' in input3):
            model_data_train,model_data_test = deal_data(model_data,'TN')
            forecast = train_model(model_data_train,model_data_test)
            data = float(forecast[forecast['ds']=='2022-3-19']['yhat'])
            end_data = get_endData(forecast,input2)
            answer ='In {} on {}, {} will be {:.2f}.'.format(input1,str(input2).replace("2022","2023"),"the lowest temperature",end_data)

        elif('average' in input3):
            model_data_train,model_data_test = deal_data(model_data,'TG')
            forecast = train_model(model_data_train,model_data_test)
            end_data = get_endData(forecast,input2)
            # print(end_data)
            answer ='In {} on {}, {} will be {:.2f}.'.format(input1,str(input2).replace("2022","2023"),"the average temperature",end_data)
        else:
            model_test('edinburgh',input2,'high')
            model_test('edinburgh',input2,'low')
            model_test('edinburgh',input2,'average')
    elif('london' in input1):
        if('high' in  input3):
            model_data_train,model_data_test = deal_data(model_data_BARINDISI,'TX')
            forecast = train_model(model_data_train,model_data_test)
            end_data = get_endData(forecast,input2)
            # print(end_data)
            answer ='In {} on {}, {} will be {:.2f}.'.format(input1,str(input2).replace("2020","2023"),"the highest temperature",end_data)

        elif('low' in input3):
            model_data_train,model_data_test = deal_data(model_data_BARINDISI,'TN')
            forecast = train_model(model_data_train,model_data_test)
            end_data = get_endData(forecast,input2)
            # print(end_data)
            answer ='In {} on {}, {} will be {:.2f}.'.format(input1,str(input2).replace("2020","2023"),"the lowest temperature", end_data)

        elif('average' in input3):
            model_data_train,model_data_test = deal_data(model_data_BARINDISI,'TG')
            forecast = train_model(model_data_train,model_data_test)
            end_data = get_endData(forecast,input2)
            # print(end_data)
            answer ='In {} on {}, {} will be {:.2f}.'.format(input1,str(input2).replace("2020","2023"),"the average temperature",end_data)
        else:
            model_test('london',input2,'high')
            model_test('london',input2,'low')
            model_test('london',input2,'average')
    print(answer)




'''
main body
'''
from datetime import datetime, timedelta
now = datetime.now()# capture the  date

input1 = ''
while input1 !='q':
    input1 = input("\nHello,I am Oracle.May I ask if you would like to inquire about the temperature in Edinburgh or London？If you want to terminate the program, please enter q.\nPlease input content:").lower()
    if('edinburgh' in input1):
        print("\nYou have successfully selected a city:{}. Please continue.".format("edinburgh"))
        flag = True
        while(flag):
            input2 = ''
            input2 = input("\nMay I ask which day`s temperature you would like to inquire about? (Only a specific date within the next seven days, eg.2023-3-21,this week,recently,tomorrow,the day after tomorrow)\nPlease input content:").lower()
            if( 'the day after tomorrow' in input2):
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                # calculate the date of the day after tomorrow
                two_days_later = now + timedelta(days=2)
                two_days_later_str = two_days_later.strftime("%Y-%m-%d")
                two_days_later_str  = two_days_later_str.replace('2023','2022')

                if("high" in input3):
                    model_test('edinburgh',two_days_later_str,'high')
                    flag = False
                elif("low" in input3):
                    model_test('edinburgh',two_days_later_str,'low')
                    flag = False
                elif("average" in input3):
                    model_test('edinburgh',two_days_later_str,'average')
                    flag = False
                else:
                    model_test('edinburgh',two_days_later_str,'')
                    flag = False
            elif('tomorrow' in input2  ):
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                # calculate the date of tomorrow
                tomorrow = now + timedelta(days=1)
                tomorrow_str = tomorrow.strftime("%Y-%m-%d")
                tomorrow_str = tomorrow_str.replace('2023','2022')

                if("high" in input3):
                    model_test('edinburgh',tomorrow_str,'high')
                    flag = False
                elif("low" in input3):
                    model_test('edinburgh',tomorrow_str,'low')
                    flag = False
                elif("average" in input3):
                    model_test('edinburgh',tomorrow_str,'average')
                    flag = False
                else:
                    model_test('edinburgh',tomorrow_str,'')
                    flag = False
            elif( 'recently' in input2):
                #calculate the dates for the next 3 days
                future_dates3 = []
                for i in range(1, 4):
                    date = now + timedelta(days=i)
                    date_str = date.strftime("%Y-%m-%d")
                    date_str  = date_str.replace('2023','2022')
                    future_dates3.append(date_str)
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                if("high" in input3):
                    for day in future_dates3:
                        model_test('edinburgh',day,'high')
                    flag = False
                elif("low" in input3):
                    for day in future_dates3:
                        model_test('edinburgh',day,'low')
                    flag = False
                elif("average" in input3):
                    for day in future_dates3:
                        model_test('edinburgh',day,'average')
                    flag = False
                else:
                    for day in future_dates3:
                        model_test('edinburgh',day,'')
                    flag = False
            elif( 'this week' in input2):
                
                # calculate the dates for the next 7 days
                future_dates7 = []
                for i in range(1, 8):
                    date = now + timedelta(days=i)
                    date_str = date.strftime("%Y-%m-%d")
                    date_str  = date_str.replace('2023','2022')
                    future_dates7.append(date_str)
                    
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                if("high" in input3):
                    for day in future_dates7:
                        model_test('edinburgh',day,'high')
                    flag = False
                elif("low" in input3):
                    for day in future_dates7:
                        model_test('edinburgh',day,'low')
                    flag = False
                elif("average" in input3):
                    for day in future_dates7:
                        model_test('edinburgh',day,'average')
                    flag = False
                else:
                    for day in future_dates7:
                        model_test('edinburgh',day,'')
                    flag = False
            elif('-' in input2 and len(input2.split("-")) > 2 and input2.split("-")[0] == '2023' and int(input2.split("-")[1]) >0 and int(input2.split("-")[1])<13 and int(input2.split("-")[2])>0 and int(input2.split("-")[2])<32):
                input2 = input2.replace('2023','2022') 
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                if("high" in input3):
                    model_test('edinburgh',input2,'average')
                    flag = False
                elif("low" in input3):
                    model_test('edinburgh',input2,'low')
                    flag = False
                elif("average" in input3):
                    model_test('edinburgh',input2,'average')
                    flag = False
                else:
                    model_test('edinburgh',input2,'')
                    flag = False
            else:
                print(" Out of forecast range, please re-enter the date as required.")


    elif('london' in input1):
        print("\nYou have successfully selected a city:{}. Please continue.".format("london"))
        flag = True
        while(flag):
            input2 = input("\nMay i ask which day`s temperature you would like to inquire about? (Only a specificdate within the next seven days, eg.,this week,recently,tomorrow,the day after tomorrow)\nPlease input content:").lower()
            if( 'the day after tomorrow' in input2):
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                # get the date of the day after tomorrow
                two_days_later = now + timedelta(days=2)
                two_days_later_str = two_days_later.strftime("%Y-%m-%d")
                two_days_later_str  = two_days_later_str.replace('2023','2020')

                if("high" in input3):
                    model_test('london',two_days_later_str,'high')
                    flag = False
                elif("low" in input3):
                    model_test('london',two_days_later_str,'low')
                    flag = False
                elif("average" in input3):
                    model_test('london',two_days_later_str,'average')
                    flag = False
                else:
                    model_test('london',two_days_later_str,'')
                    flag = False
            elif('tomorrow' in input2  ):
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                # calculate the date of tomorrow
                tomorrow = now + timedelta(days=1)
                tomorrow_str = tomorrow.strftime("%Y-%m-%d")
                tomorrow_str = tomorrow_str.replace('2023','2020')

                if("high" in input3):
                    model_test('london',tomorrow_str,'high')
                    flag = False
                elif("low" in input3):
                    model_test('london',tomorrow_str,'low')
                    flag = False
                elif("average" in input3):
                    model_test('london',tomorrow_str,'average')
                    flag = False
                else:
                    model_test('london',tomorrow_str,'')
                    flag = False
            elif( 'recently' in input2):
                # calculate the date for next 3 days
                future_dates3 = []
                for i in range(1, 4):
                    date = now + timedelta(days=i)
                    date_str = date.strftime("%Y-%m-%d")
                    date_str  = date_str.replace('2023','2020')
                    future_dates3.append(date_str)
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                if("high" in input3):
                    for day in future_dates3:
                        model_test('london',day,'high')
                    flag = False
                elif("low" in input3):
                    for day in future_dates3:
                        model_test('london',day,'low')
                    flag = False
                elif("average" in input3):
                    for day in future_dates3:
                        model_test('london',day,'average')
                    flag = False
                else:
                    for day in future_dates3:
                        model_test('london',day,'')
                    flag = False
            elif( 'this week' in input2):
                
                # calculate the date for next 7 days
                future_dates7 = []
                for i in range(1, 8):
                    date = now + timedelta(days=i)
                    date_str = date.strftime("%Y-%m-%d")
                    date_str  = date_str.replace('2023','2020')
                    future_dates7.append(date_str)
                    
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                if("high" in input3):
                    for day in future_dates7:
                        model_test('london',day,'high')
                    flag = False
                elif("low" in input3):
                    for day in future_dates7:
                        model_test('london',day,'low')
                    flag = False
                elif("average" in input3):
                    for day in future_dates7:
                        model_test('london',day,'average')
                    flag = False
                else:
                    for day in future_dates7:
                        model_test('london',day,'')
                    flag = False
            elif('-' in input2 and len(input2.split("-")) > 2 and input2.split("-")[0] == '2023' and int(input2.split("-")[1]) >0 and int(input2.split("-")[1])<13 and int(input2.split("-")[2])>0 and int(input2.split("-")[2])<32):
                input2 = input2.replace('2023','2020')
                print("\nI got it! Could you please tell me the specific type of temperature that you want to know?  Is it the high temperature, the  low temperature, or the average temperature?")
                input3 = input("Type in here, please (high,low,average or all): ").lower()
                print('\n')
                if("high" in input3):
                    model_test('london',input2,'average')
                    flag = False
                elif("low" in input3):
                    model_test('london',input2,'low')
                    flag = False
                elif("average" in input3):
                    model_test('london',input2,'average')
                    flag = False
                else:
                    model_test('london',input2,'')
                    flag = False
            else:
                print(" Out of forecast range, please re-enter the date as required.")
    else:
        print("\nWe only offer temperatures in Edinburgh and London.Please re-enter the city you are looking for.\n")
