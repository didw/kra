# -*- coding:utf-8 -*-
import re
import pandas as pd
import numpy as np
import datetime
import os.path
import glob
from sklearn.externals import joblib

DEBUG = False
def make_one_hot(name):
    jockey_list = ["볼스터", "조한별", "후지이", "페로비치", "설동복", "고성이", "아담", "무타자", "이종섭", "하마다", "쉴렉터", "윤대근", "다실바", "파둘", "조인권", "차베스", "오카베", "스탠리", "이쿠야스", "윤태혁", "이용호", "최범현", "이성재", "윤기정", "하밋", "리오", "손영진", "유광희", "임성실", "임기원", "김옥성", "지하주", "하정훈", "조창욱", "김철호", "우에노", "니콜라", "윤영민", "안토니", "천창기", "요시로", "박병윤", "티턴", "임대규", "아오키", "박금만", "페르난도", "전덕용", "이애리", "신형철", "토시코", "서승운", "송석헌", "비라펜", "이현종", "마틴", "유상완", "이준동", "일디림", "보렐리", "심승태", "구영준", "오경환", "김도현", "서도수", "양희진", "신대전", "이희천", "노엘", "스티븐", "요시하라", "박종미", "이기회", "니알", "맥네일", "커티스", "정현", "다카하시", "이동하", "함완식", "김태훈", "우찌다", "정정희", "무찌", "후마", "이성환", "이동국", "허재영", "괘칸", "이준철", "부민호", "대니", "에디", "최정섭", "드웨인", "임란", "김태경", "쯔바사", "박을운", "유현명", "마호멧", "이상혁", "방춘식", "마이", "송경윤", "황영원", "이아나", "이기웅", "문정균", "박종현", "김효섭", "이강서", "게릿", "김도중", "김태준", "유재필", "사무엘", "누네스", "김용근", "문세영", "얀", "한성열", "이세이", "박상우", "양영남", "이쿠", "유미라", "한창민", "유승완", "레이몬", "오셰이", "아카네", "리챠드", "씨즈용", "김귀배", "김영민", "문중원", "가토", "김동영", "미유키", "마토바", "딘홀랜드", "다나카", "테일러", "황종우", "마코토", "크레이그", "웨인", "김낙현", "김민수", "이해동", "코스케", "호반", "김명신", "산토스", "최이모", "야노", "김동민", "황순도", "한상규", "라케쉬", "이금주", "박시천", "김석봉", "제임스", "타츠시로", "송재철", "최원준", "데이빗", "신지", "홀랜드", "안토니오", "호세", "파올로", "권석원", "이정선", "로날드", "정기용", "람호이", "수쿤벵", "신이치", "안병기", "이찬호", "두소", "요시다", "정평수", "원정일", "조재로", "이효식", "올리버", "조경호", "조찬훈", "최봉주", "사토시", "벨리", "노조무", "이철경", "김어수", "쿠니", "로버트", "재크", "케빈", "유셀", "패트릭", "티탄", "김혜선", "아베", "그레고리", "김정준", "제롬", "히토미", "박수홍", "박태종", "안효리", "조희원", "박현우", "김남성", "이시바시", "에이키", "장추열", "마시마", "아즈하", "이혁", "이신영", "타케히로", "웡친추엔", "파스토", "로널드", "오야마", "정동철", "루이스", "카시와", "김동철", "스캇", "로리", "테츠야", "히라세", "알도", "아킨", "잭슨", "조슈아", "조성곤", "김영진", "아흐메", "채규준", "김혜성", "채상현", "우창구", "김동균", "다니엘", "김동수", "칼슨", "우이스", "웡캄총", "밍완테", "최시대"]
    res = np.zeros(len(jockey_list))
    for i in range(len(jockey_list)):
        if name == jockey_list[i]:
            res[i] = 1
    return res

def get_jockey(name):
    return make_one_hot(name)


if __name__ == '__main__':
    print(len(get_jockey("아담")))

