'''
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex
Age	Age in years
sibsp	# of siblings / spouses aboard the Titanic
parch	# of parents / children aboard the Titanic
ticket	Ticket number
fare	Passenger fare
cabin	Cabin number
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton


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
        print('--------------------------------1. ------------------------------------')
        print('--------------------------------2. ------------------------------------')
        print('--------------------------------3. ------------------------------------')

