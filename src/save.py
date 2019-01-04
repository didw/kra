import re
import datetime
import requests
import pymysql
import json

with open('src/mysql_key.json') as f:
    mysql_key = json.load(f)

mydb = pymysql.connect(host='kra-rds-mysql.cbc9afwyti4d.ap-northeast-2.rds.amazonaws.com', user=mysql_key['user'], password=mysql_key['passwd'], db='kra', charset='utf8')

mycursor = mydb.cursor()

def update_horse(update_date):
    update_date_i = update_date.strftime("%Y%m%d")
    test_url = 'http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/{}sdb1.txt&meet=1'.format(update_date_i)

    r = requests.get(test_url)
    if len(r.text) < 10:
        print("pass {}".format(test_url))
        return
    for line in r.text.split('\r\n'):
        if len(line) == 0 or line[0]=='-' or line[:2]=='마명':
            continue

        # name
        p = re.compile(r'[가-힣]+')
        m = p.match(line)
        try:
            name = m.group()
            line = line[m.end():].strip()
            #print(name)
        except Exception as e:
            print(e, line)
            pass
        
        # hometown
        p = re.compile(r'[가-힣()]+')
        m = p.match(line)
        try:
            hometown = m.group()
            line = line[m.end():].strip()
            #print(hometown)
        except Exception as e:
            print(e, line)
            pass
        
        # gender
        p = re.compile(r'[가-힣]')
        m = p.match(line)
        try:
            gender = m.group()
            line = line[m.end():].strip()
            #print(gender)
        except Exception as e:
            print(e, line)
            pass
        
        # birthdate
        p = re.compile(r'\d{4}/\d{2}/\d{2}')
        m = p.match(line)
        try:
            birthdate = m.group()
            line = line[m.end():].strip()
            #print(birthdate)
        except Exception as e:
            print(e, line)
            pass
        
        # age
        p = re.compile(r'\d')
        m = p.match(line)
        try:
            age = int(m.group())
            line = line[m.end():].strip()
            #print(age)
        except Exception as e:
            print(e, line)
            pass
        
        # grade
        p = re.compile(r'[가-힣\d ]{2}')
        m = p.match(line)
        try:
            grade = m.group()
            line = line[m.end():].strip()
            #print(grade)
        except Exception as e:
            print(e, line)
            pass
        
        # group
        p = re.compile(r'\d{2}')
        m = p.match(line)
        try:
            group = m.group()
            line = line[m.end():].strip()
            #print(group)
        except Exception as e:
            print(e, line)
            pass
        
        # trainer
        p = re.compile(r'[가-힣]{3}')
        m = p.match(line)
        try:
            trainer = m.group()
            line = line[m.end():].strip()
            #print(trainer)
        except Exception as e:
            print(e, line)
            pass
        
        # owner
        p = re.compile(r'[가-힣()a-z\d]+')
        m = p.match(line)
        try:
            owner = m.group()
            line = line[m.end():].strip()
            #print(owner)
        except Exception as e:
            print(e, line)
            pass
        
        # father
        p = re.compile(r'[가-힣a-zA-ZⅡ`\. ]+?  ')
        m = p.match(line)
        try:
            father = m.group().strip()
            line = line[m.end():].strip()
            #print(father)
        except Exception as e:
            print(e, line)
            pass
        
        # mother
        p = re.compile(r'[가-힣a-zA-ZⅡ`\. ]+  ')
        m = p.match(line)
        try:
            mother = m.group().strip()
            line = line[m.end():].strip()
            #print(mother)
        except Exception as e:
            print(e, line)
            pass
        
        # total_participate
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_participate = int(m.group())
            line = line[m.end():].strip()
            #print(total_participate)
        except Exception as e:
            print(e, line)
            pass
        
        # total_first
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_first = int(m.group())
            line = line[m.end():].strip()
            #print(total_first)
        except Exception as e:
            print(e, line)
            pass
        
        # total_second
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_second = int(m.group())
            line = line[m.end():].strip()
            #print(total_second)
        except Exception as e:
            print(e, line)
            pass
        
        # total_third
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_third = int(int(m.group()))
            line = line[m.end():].strip()
            #print(total_third)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1year
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1year = int(m.group())
            line = line[m.end():].strip()
            #print(total_1year)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1y_first
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1y_first = int(m.group())
            line = line[m.end():].strip()
            #print(total_1y_first)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1y_second
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1y_second = int(m.group())
            line = line[m.end():].strip()
            #print(total_1y_second)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1y_third
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1y_third = int(m.group())
            line = line[m.end():].strip()
            #print(total_1y_third)
        except Exception as e:
            print(e, line)
            pass
        
        # total_prize
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_prize = int(m.group())
            line = line[m.end():]
            #print(total_prize)
        except Exception as e:
            print(e, line)
            pass
        
        # rating
        p = re.compile(r'.*[\d{2} ]  ')
        m = p.match(line)
        try:
            rating = int(m.group())
            line = line[m.end():].strip()
            #print(rating)
        except Exception as e:
            print(e, "line:{}".format(line))
            pass
        
        # recent_price
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            recent_price = int(m.group())
            line = line[m.end():].strip()
            #print(recent_price)
        except Exception as e:
            print(e, "line:{}".format(line))
            pass
        

        sql = "INSERT INTO `horse` (`name`, `hometown`, `gender`, `birthdate`, `age`, `grade`, `group`, `trainer`, `owner`, \
            `father`, `mother`, `total_participate`, `total_first`, `total_second`, `total_third`, \
            `total_1year`, `total_1y_first`, `total_1y_second`, `total_1y_third`, `total_prize`, `rating`, \
            `recent_price`, `update_date`) VALUES \
            (\'{}\', \'{}\', \'{}\', \'{}\', {}, \'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \
             \'{}\', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, \'{}\');"
        sql = sql.format(name, hometown, gender, birthdate, age, grade, group, trainer, owner, father, 
            mother, total_participate, total_first, total_second, total_third, total_1year, 
            total_1y_first, total_1y_second, total_1y_third, total_prize, rating, recent_price, update_date)
        print(sql)
        mycursor.execute(sql)
        mydb.commit()
        #print(name, hometown, gender, birthdate, age, grade, group, trainer, owner, father, mother, total_participate, total_first, total_second, total_third,
        #    total_1year, total_1y_first, total_1y_second, total_1y_third, total_prize, rating, recent_price, update_date)

    print("updated {}".format(update_date))

if __name__ == '__main__':
    cur_date = datetime.datetime(2018,12,10)
    while True:
        cur_date += datetime.timedelta(days=1)
        if cur_date > datetime.datetime(2018,12,15):
            break
        update_horse(cur_date)
