import re

import requests

def update_horse(update_date):
    test_url = 'http://race.kra.co.kr/dbdata/fileDownLoad.do?fn=internet/seoul/horse/{}sdb1.txt&meet=1'.format(update_date)

    r = requests.get(test_url)
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
            age = m.group()
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
        p = re.compile(r'[가-힣a-zA-ZⅡ\'\. ]+?  ')
        m = p.match(line)
        try:
            father = m.group().strip()
            line = line[m.end():].strip()
            #print(father)
        except Exception as e:
            print(e, line)
            pass
        
        # mother
        p = re.compile(r'[가-힣a-zA-ZⅡ\'\. ]+  ')
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
            total_participate = m.group()
            line = line[m.end():].strip()
            #print(total_participate)
        except Exception as e:
            print(e, line)
            pass
        
        # total_first
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_first = m.group()
            line = line[m.end():].strip()
            #print(total_first)
        except Exception as e:
            print(e, line)
            pass
        
        # total_second
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_second = m.group()
            line = line[m.end():].strip()
            #print(total_second)
        except Exception as e:
            print(e, line)
            pass
        
        # total_third
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_third = m.group()
            line = line[m.end():].strip()
            #print(total_third)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1year
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1year = m.group()
            line = line[m.end():].strip()
            #print(total_1year)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1y_first
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1y_first = m.group()
            line = line[m.end():].strip()
            #print(total_1y_first)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1y_second
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1y_second = m.group()
            line = line[m.end():].strip()
            #print(total_1y_second)
        except Exception as e:
            print(e, line)
            pass
        
        # total_1y_third
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_1y_third = m.group()
            line = line[m.end():].strip()
            #print(total_1y_third)
        except Exception as e:
            print(e, line)
            pass
        
        # total_prize
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            total_prize = m.group()
            line = line[m.end():]
            #print(total_prize)
        except Exception as e:
            print(e, line)
            pass
        
        # rating
        p = re.compile(r'.*[\d{2} ]  ')
        m = p.match(line)
        try:
            rating = m.group()
            line = line[m.end():].strip()
            #print(rating)
        except Exception as e:
            print(e, "line:{}".format(line))
            pass
        
        # recent_price
        p = re.compile(r'\d+')
        m = p.match(line)
        try:
            recent_price = m.group()
            line = line[m.end():].strip()
            #print(recent_price)
        except Exception as e:
            print(e, "line:{}".format(line))
            pass
        
        
        print(name, hometown, gender, birthdate, age, grade, group, trainer, owner, father, mother, total_participate, total_first, total_second, total_third,
            total_1year, total_1y_first, total_1y_second, total_1y_third, total_prize, rating, recent_price, update_date)
        
