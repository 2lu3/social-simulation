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
        self.buying_failed_number = 0
        self.ideal_mask_number = random() * 100 + 50
        self.posessing_mask_number = round(random() * 150)
        self.blacklist = []
        self.wanted_mask_number = 0

    def agt_step(self):
        distance = 15 + self.wanted_mask_number / 10

        if self.wanted_mask_number == 0:
            distance = -1

        near_stores = set()
        for x in [ - self.universe.wide, 0, self.universe.wide]:
            for y in [-self.universe.height, 0, self.universe.height]:
                ([near_stores.add(store) for store in self.universe.store_list if sqrt((store.x + x - self.x)**2 + (store.y + y - self.y)**2) < distance])

        if self.is_application_user == True:

            if len(near_stores) > 0:
                weights = [store.posessing_mask_number for store in near_stores]
                # 在庫がすべて0のとき
                if sum(weights) <= 0:
                    weights = [1 for _ in range(len(weights))]
                applying_store = choices(list(near_stores), weights=weights)[0]
            else:
                applying_store = None
        else:
            if len(set(near_stores) - set(self.blacklist)) > 0:
                applying_store = choice(list(set(near_stores) - set(self.blacklist)))
            elif len(set(near_stores)) > 0:
                applying_store = choice(list(near_stores))
            else:
                applying_store = None

        if applying_store != None:
            # self.wanted_mask_number を50の倍数に
            mask_number = 50
            self.wanted_mask_number = np.ceil(self.wanted_mask_number / mask_number) * mask_number
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
        #self.initial_mask_number = universe.all_supply / universe.store_number
        self.posessing_mask_number = self.initial_mask_number

    def agt_init(self):
        self.request_information = []
        self.bought_times = 0
        self.time = randint(0, self.universe.mask_storage_period-1)

    def agt_step(self):
        pass


    def summarize_requests(self):
        shuffle(self.request_information)
        self.bought_times = 0
        self.bought_number = 0
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
                self.bought_number += self.posessing_mask_number
                self.bought_times += 1
                self.posessing_mask_number = 0
                consumer.buying_failed_number = 0
            else:
                # 店の在庫が十分にある場合
                consumer.posessing_mask_number += applying_mask_number
                self.bought_number += applying_mask_number
                self.bought_times += 1
                self.posessing_mask_number -= applying_mask_number
                consumer.buying_failed_number = 0
                if self in consumer.blacklist:
                    consumer.blacklist.remove(self)
        if self.bought_times != 0:
            self.bought_number /= self.bought_times

class Universe:
    def __init__(self):
        self.consumer_list = []
        self.store_list = []
        self.wide = 50
        self.height = 50
        self.statistics_df = pd.DataFrame(columns=['mask mean', 'mask std', 'wanted mean'])
        self.raw_df = pd.DataFrame()
        self.step = 0

    def univ_init(self, store_number=10, consumer_number=100, buying_number_limit=-1, mask_storage_period=10, application_percentage = 0., max_store_consumer_distance=15, supply_portion=1.0, consume_mask_number=2.5):

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
        self.wanted_mask_number_std = []
        self.store_mask_average = []
        self.store_mask_std = []
        self.store_bought_num_average = []
        self.store_bought_times_average = []

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
            if consumer.x < 0:
                consumer.x += self.wide
            if consumer.x > self.wide:
                consumer.x -= self.wide
            consumer.y = store.y + r * sin(degree)
            if consumer.y < 0:
                consumer.y += self.height
            if consumer.y >  self.height:
                consumer.y -= self.height



        # アプリ利用者を決める
        application_users_num = round(self.application_portion / 100 * self.consumer_number)
        application_users = sample(self.consumer_list, application_users_num)
        for user in application_users:
            user.is_application_user = True


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
        store_bought_number = [store.bought_number for store in self.store_list]
        store_bought_times = [store.bought_times for store in self.store_list]

        self.mask_average.append(np.average(consumer_mask_list))
        self.mask_std.append(np.std(consumer_mask_list))
        self.wanted_mask_number_average.append( np.average(wanted_mask_list))
        self.wanted_mask_number_std.append(np.std(wanted_mask_list))
        self.store_mask_average.append(np.average(store_mask_list))
        self.store_mask_std.append(np.std(store_mask_list))
        self.store_bought_num_average.append(np.average(store_bought_number))
        self.store_bought_times_average.append(np.average(store_bought_times))
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
        for data in [self.mask_average, self.mask_std, self.wanted_mask_number_average]:
        #for data in [self.mask_average, self.mask_std, self.wanted_mask_number_average, self.store_mask_average, self.store_bought_num_average, self.store_bought_times_average]:
            plt.plot([i for i in range(len(data))] , data)
        #plt.legend(['mask mean', 'mask std', 'wanted mean'])
        #plt.legend(['consumer posessing mask', 'consume posessing mask std', 'consumer wanted mask mean', 'store mask mean', 'store bought times mean', 'store bought num mean'])
        plt.legend(['consumer posessing mask', 'consume posessing mask std', 'consumer wanted mask mean'] )
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
    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax2 = fig.add_subplot(211)


    trial_num = 3
    # 片側にmoving_average
    moving_average =  5
    record_steps = [50, 100, 150]
    conditions = []
    #for supply_portion in [1.1]:
    #    for buying_number_limit in [50]:
    #        for application_percentage in [0]:
    for supply_portion in [0.9, 1.0, 1.05]:
        for buying_number_limit in [50, 100, 150, 1000]:
            for application_percentage in [0, 25, 50, 75, 100]:
                conditions.append({
                    "supply_portion": supply_portion,
                    "buying_number_limit": buying_number_limit,
                    "application_percentage": application_percentage,
                    })
    df_result = [pd.DataFrame(columns=['supply_portion', 'buying_number_limit', 'application_percentage' ,'consumer mask mean', 'consumer mask std', 'consumer wanted mean', 'consumer wanted std', 'store mask mean', 'store mask std']) for _ in range(len(record_steps))]

    for condition in tqdm(conditions):
        supply_portion = condition['supply_portion']
        application_percentage = condition['application_percentage']
        buying_number_limit = condition['buying_number_limit']
        df_index = f'suppy{supply_portion}-limit{buying_number_limit}-app{application_percentage}'


        result = np.zeros((len(df_result), len(df_result[0].columns)))
        for _ in range(trial_num):
            universe = Universe()
            universe.univ_init(consumer_number=1000, store_number= 10, supply_portion=supply_portion,mask_storage_period=5, application_percentage=application_percentage, buying_number_limit=buying_number_limit)

            for consumer in universe.consumer_list:
                consumer.agt_init()
            for store in universe.store_list:
                store.agt_init()
            for step in range(record_steps[-1] + moving_average +1):
                for index, record_step in enumerate(record_steps):
                    if step == record_step + moving_average // 2:
                        result[index, :] += np.array([
                            supply_portion,
                            buying_number_limit,
                            application_percentage,
                            np.average(universe.mask_average[-moving_average:]),
                            np.average(universe.mask_std[-moving_average:]),
                            np.average(universe.wanted_mask_number_average[-moving_average:]),
                            np.average(universe.wanted_mask_number_std[-moving_average:]),
                            np.average(universe.store_mask_average[-moving_average:]),
                            np.average(universe.store_mask_std[-moving_average:])])
                universe.univ_step_begin()
                for consumer in universe.consumer_list:
                    consumer.agt_step()
                for store in universe.store_list:
                    store.agt_step()
                universe.univ_step_end()

#                    for consumer in universe.consumer_list:
#                        ax1.scatter(consumer.x, consumer.y, c='black')
#                    for store in universe.store_list:
#                        ax1.scatter(store.x, store.y, c='red')
#                    for data in [universe.mask_average, universe.mask_std, universe.store_mask_average, universe.store_mask_std, universe.wanted_mask_number_average, universe.wanted_mask_number_std]:
#                        ax2.plot([i for i in range(len(data))] , data)
#                    ax2.legend(['C posessing mask', 'C posessing mask std', 'S mask mean', 'S mask std', 'C wanted mean', 'C wanted std'] )
#                    ax2.grid()
#                    fig.savefig('data/' + df_index+ '.png')
#                    plt.cla()

        result /= trial_num

        for i in range(len(df_result)):
            df_result[i] = df_result[i].append(pd.Series(result[i], index=df_result[i].columns), ignore_index=True)
    for i, df in enumerate(df_result):
        df.to_csv(f'data/result{record_steps[i]}.csv', header=True, index=False)



if __name__ == '__main__':
    main()
