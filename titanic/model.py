'''
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex
Age	Age in years
sibsp	# of siblings / spouses aboard the Titanic 동반한 형제,자매,배우자
parch	# of parents / children aboard the Titanic 동반환 부모,자식
ticket	Ticket number
fare	Passenger fare
cabin	Cabin number 객실번호
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

'''
import pandas as pd
import numpy as np
class TitanicModel:
    def __init__(self):
        self._context =None
        self._fname =None
        self._train =None
        self._test =None
        self._test_id =None

    @property
    def context(self) -> object:return self._context

    @context.setter
    def context(self,context):self._context=context

    @property
    def fname(self) -> object: return self._fname

    @fname.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> object: return self._train

    @train.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test

    @test.setter
    def test(self, test): self._test = test

    @property
    def test_id(self) -> object: return self._test_id

    @test_id.setter
    def test_id(self, test_id): self._test_id = test_id


    def new_file(self) -> str : return self._context + self._fname

    def new_dfame(self)-> object:
        file = self.new_file()
        return pd.read_csv(file)

    def hook_process(self,train,test)->object:
        print('--------------------------------1. Cabin Ticket삭제------------------------------------')

        t = self.drop_feature(train,test,'Cabin')
        t = self.drop_feature(t[0],t[1],'Ticket')
        print('--------------------------------2. embarked 승선한 항구명 편집:문자열을 숫자로------------------------------------')
        t= self.embarked_norminal(t[0],t[1])
        print('--------------------------------3. 신분 편집:문자열을 숫자로------------------------------------')
        t=self.title_norminal(t[0],t[1])
        print('--------------------------------4. Name,PassingerId삭제------------------------------------')

        t = self.drop_feature(t[0], t[1], 'Name')
        #테스트 프로세스에 사용하기위해 남겨둔다. 즉 정확도

        self._test_id = test['PassengerId']
        t = self.drop_feature(t[0], t[1], 'PassengerId')
        print('--------------------------------5. Age 편집------------------------------------')
        t=self.age_ordinal(t[0],t[1])

        print('--------------------------------6. Fare 편집------------------------------------')
        t = self.fare_ordinal(t[0], t[1])
        print('--------------------------------7. Fare 편집------------------------------------')
        #Fare편집이 끝났음으로 Fare컬럼 삭제
        t = self.drop_feature(t[0], t[1], 'Fare')
        print('--------------------------------8. Sex normial 편집------------------------------------')
        t = self.sex_nominal(t[0], t[1])
        t[1] = t[1].fillna({'FareBand':1})
        a = self.null_sum(t[1])
        print('널의 수량 {} 개'.format(a))
        self._test =t[1]
        return t[0]

    @staticmethod
    def null_sum(train)-> int:
        return train.isnull().sum()

    @staticmethod
    def drop_feature(train,test,feature) ->[]:
        train = train.drop([feature],axis=1) # axis=1 는 컬럼을 삭제 .디폴트는 0 즉 행을 삭제
        test = test.drop([feature], axis=1)
        return [train,test]

    #승선위치를 숫자화
    @staticmethod
    def embarked_norminal(train,test) -> []:
        #c_city = train[train['Embarked'] == 'C'].shape[0]
        #s_city = train[train['Embarked'] == 'S'].shape[0]
        #q_city = train[train['Embarked'] == 'Q'].shape[0]

        train = train.fillna({'Embarked':'S'})
        city_mapping = {'S': 1, 'C': 2, 'Q':3}
        train['Embarked'] = train['Embarked'].map(city_mapping)
        test['Embarked'] = test['Embarked'].map(city_mapping)
        return [train, test]

    @staticmethod
    def title_norminal(train,test) -> []:
        combine =[train,test]
        for dataset in combine:
            #신분에 따른 생졸율을 알아내기 위해 . 앞에 신분을 나타내는 문자열 뽑기
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)

        for dataset in combine:
            dataset['Title'] \
                =  dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkher','Dona'],'Rare')

            dataset['Title'] \
                = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

            dataset['Title'] \
                = dataset['Title'].replace(['Mile', 'Ms'], 'Miss')

        #print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

        train[['Title','Survived']].groupby(['Title'],as_index=False).mean()
        title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Rare':6,'Mne':7}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        return [train,test]

    @staticmethod
    def sex_nominal(train,test) -> []:
        combine = [train, test]
        sex_mapping = {'male':0,'female':1}
        for dataset in combine:
            dataset['Sex'] =  dataset['Sex'].map(sex_mapping)

        return [train,test]

    @staticmethod
    def age_ordinal(train, test) -> []:
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        #연령대별로 나누기
        bins=[-1,0,5,12,18,24,35,60,np.inf]
        labels=['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

        train['AgeGroup'] = pd.cut(train['Age'],bins,labels=labels)#cut
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)  # cut

        age_title_mapping = {0:'Unknown',1:'Baby',2:'Child',3:'Teenager',4:'Student',5:'Young Adult',6:'Adult',7:'Senior'}
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]

        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]


        age_mapping = {'Unknown':0,'Baby':1,'Child':2,'Teenager':3,'Student':4,'Young Adult':5,'Adult':6,'Senior':7}
        train['AgeGroup'] =train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        print(train['AgeGroup'].head())
        return [train, test]

    @staticmethod
    def fare_ordinal(train,test) ->[]:
        train['FareBand'] =pd.qcut(train['Fare'],4,labels={1,2,3,4})
        test['FareBand'] = pd.qcut(test['Fare'], 4, labels={1, 2, 3, 4})
        return [train, test]