from random import seed, gauss, choices, random, sample, choice, randint, shuffle
from math import sqrt, pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class Consumer:
    def __init__(self, universe):
        self.x = 0.
        self.y = 0.
        self.universe = universe

        self.is_application_user = False

    def agt_init(self):
        self.wanted_mask_number = 0
        self.buying_failed_number = 0
        self.ideal_mask_number = random() * 100 + 50
        self.posessing_mask_number = self.ideal_mask_number
        self.blacklist = []

    def agt_step(self):
        distance = 15 + self.wanted_mask_number / 10

        if self.wanted_mask_number == 0:
            distance = -1

        near_stores = [store for store in self.universe.store_list if sqrt((store.x - self.x)**2 + (store.y - self.y)**2) < distance]
        print(len(near_stores))

        if self.is_application_user == True:
            if len(near_stores):
                weights = [store.posessing_mask_number for store in near_stores]
                applying_store = choices(near_stores, weights=weights)[0]
            else:
                applying_store = None
        else:
            if len(set(near_stores) - set(self.blacklist)) > 0:
                applying_store = choice(list(set(near_stores) - set(self.blacklist)))
            else:
                applying_store = None

        if applying_store != None:
            # self.wanted_mask_number を50の倍数に
            mask_number = 10
            self.wanted_mask_number = mask_number* (self.wanted_mask_number // mask_number)
            if self.universe.buying_number_limit == -1:
                applying_store.request_information.append([self, self.wanted_mask_number])
            else:
                applying_store.request_information.append([self, min(self.wanted_mask_number, self.universe.buying_number_limit)])


    def feedback(self):
        self.posessing_mask_number -= self.universe.consume_mask_number 
        if self.posessing_mask_number < 0:
            self.posessing_mask_number = 0

        # 欲しいマスク枚数を更新
        self.wanted_mask_number = self.ideal_mask_number - self.posessing_mask_number
        if self.buying_failed_number != 0:
            self.wanted_mask_number += (10-5/self.buying_failed_number)**2
        if self.wanted_mask_number < 0:
            self.wanted_mask_number = 0

class Store:
    def __init__(self, universe):
        self.x = 0.
        self.y = 0.

        self.universe = universe
        self.initial_mask_number = round(universe.all_supply / universe.store_number * max(gauss(1, 0.2), 0.1))
        self.posessing_mask_number = self.initial_mask_number

    def agt_init(self):
        self.request_information = []
        self.time = randint(0, self.universe.mask_storage_period-1)

    def agt_step(self):
        pass


    def summarize_requests(self):
        shuffle(self.request_information)
        while len(self.request_information) > 0:
            [consumer, applying_mask_number] = self.request_information.pop()

            if self.posessing_mask_number == 0:
                # マスクの在庫がない場合
                consumer.buying_failed_number += 1
                consumer.blacklist.append(self)
                if len(consumer.blacklist) > 3:
                    consumer.blacklist.pop(0)
            elif self.posessing_mask_number < applying_mask_number:
                # マスクの在庫が足りない場合
                consumer.posessing_mask_number += self.posessing_mask_number
                self.posessing_mask_number = 0
                consumer.buying_failed_number = 0
            else:
                # 店の在庫が十分にある場合
                consumer.posessing_mask_number += applying_mask_number
                self.posessing_mask_number -= applying_mask_number
                consumer.buying_failed_number = 0
                if self in consumer.blacklist:
                    consumer.blacklist.remove(self)

class Universe:
    def __init__(self):
        self.consumer_list = []
        self.store_list = []
        self.wide = 50
        self.height = 50
        self.statistics_df = pd.DataFrame(columns=['mask mean', 'mask std', 'wanted mean'])
        self.raw_df = pd.DataFrame()
        self.step = 0

    def univ_init(self, store_number=10, consumer_number=100, buying_number_limit=-1, mask_storage_period=1, application_percentage = 0., max_store_consumer_distance=10, supply_portion=1.0, consume_mask_number=2.5):

        # 店の数
        self.store_number = store_number
        # 消費者数
        self.consumer_number = consumer_number
        # 購入制限. -1は制限なし
        self.buying_number_limit = buying_number_limit
        # 店に入荷する日数
        self.mask_storage_period = mask_storage_period
        # アプリ使用率
        self.application_portion = application_percentage
        # 消費量に対する供給量の割合
        self.supply_portion = supply_portion
        self.consume_mask_number = consume_mask_number

        # 統計データ
        self.mask_average = []
        self.mask_std = []
        self.wanted_mask_number_average = []
        self.store_mask_average = []

        # 1ステップあたりの全供給量(需要と供給のバランスをとる,倍率をかける余地あり)
        self.all_supply = round(self.consume_mask_number * self.consumer_number * self.mask_storage_period * self.supply_portion)

        # storeを配置
        stores = self.create_agt(Store, num=self.store_number)
        self.random_put_agtset(stores)

        # 店の初期マスク枚数の合計を求める
        self.sum_store_initial_mask_number = sum([store.initial_mask_number for store in self.store_list])

        # storeの周りにconsumerを配置
        consumers = self.create_agt(Consumer, num=self.consumer_number)
        store_selection_weights = [store.initial_mask_number for store in self.store_list]

        for consumer in consumers:
            store = choices(self.store_list, weights=store_selection_weights)[0]
            degree = random() * 2 * pi
            r = random() * max_store_consumer_distance
            consumer.x = store.x + r * cos(degree)
            consumer.y = store.y + r * sin(degree)


        # アプリ利用者を決める
        application_users_num = round(self.application_portion / 100 * self.consumer_number)
        application_users = sample(self.consumer_list, application_users_num)
        for user in application_users:
            user.is_application_user = True

        #コンソール用
        print("step数：標準偏差")

    def univ_step_begin(self):
        # マスクの在庫をふやす
        for store in self.store_list:
            if store.time == self.mask_storage_period:
                store.posessing_mask_number += self.all_supply * store.initial_mask_number / self.sum_store_initial_mask_number
                store.time = 0
            store.time += 1

    def univ_step_end(self):
        for store in self.store_list:
            store.summarize_requests()

        for consumer in self.consumer_list:
            consumer.feedback()

        consumer_mask_list = [consumer.posessing_mask_number for consumer in self.consumer_list]
        wanted_mask_list = [consumer.wanted_mask_number for consumer in self.consumer_list]
        store_mask_list = [store.posessing_mask_number for store in self.store_list]

        self.mask_average.append(np.average(consumer_mask_list))
        self.mask_std.append(np.std(consumer_mask_list))
        self.wanted_mask_number_average.append( np.average(wanted_mask_list))
        self.store_mask_average.append(np.average(store_mask_list))
        self.raw_df[self.step] = consumer_mask_list + wanted_mask_list + store_mask_list
        self.step += 1

    def univ_finish(self):
        pass


    def plot(self):
        plt.figure()
        for consumer in self.consumer_list:
            plt.scatter(consumer.x, consumer.y, c='black')
        for store in self.store_list:
            plt.scatter(store.x, store.y, c="red")
        plt.figure()
        #for data in [self.mask_average, self.mask_std, self.wanted_mask_number_average]:
        for data in [self.mask_average, self.mask_std, self.wanted_mask_number_average, self.store_mask_average]:
            plt.plot([i for i in range(len(data))] , data)
        #plt.legend(['mask mean', 'mask std', 'wanted mean'])
        plt.legend(['mask mean', 'mask std', 'wanted mean', 'store mean'])
        plt.show()

    def save_df(self, file_name):
        pass

    def create_agt(self, agt_class, num=1):
        new_agent_list = [agt_class(self) for i in range(num)]
        if agt_class == Consumer:
            self.consumer_list.extend(new_agent_list)
        elif agt_class == Store:
            self.store_list.extend(new_agent_list)
        else:
            print(agt_class, "is unknown")
        return new_agent_list
    def random_put_agtset(self, agent_list):
        for agent in agent_list:
            agent.x = random() * self.wide
            agent.y = random() * self.height



def main():
    seed(1)
    universe = Universe()
    universe.univ_init(consumer_number=20, store_number= 2, consume_mask_number=10)
    for consumer in universe.consumer_list:
        consumer.agt_init()
    for store in universe.store_list:
        store.agt_init()
    for _ in tqdm(range(100)):
        universe.univ_step_begin()
        for consumer in universe.consumer_list:
            consumer.agt_step()
        for store in universe.store_list:
            store.agt_step()
        universe.univ_step_end()

        if False:
            print(universe.step)
            for consumer in universe.consumer_list:
                print('con', consumer.posessing_mask_number)
            for store in universe.store_list:
                print('sto', store.posessing_mask_number)

    universe.plot()

if __name__ == '__main__':
    main()
