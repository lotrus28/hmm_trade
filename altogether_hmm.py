# Hi, I removed all the garbage that I only intended to use, or used in some obscure ways
# This script is as clear and compact as it can be, but still it contains some footprints of a messier version.
# E.g., cryptic returns that seem to do nothing. They used to, but they don't. And I am afraid to remove all of them,
# as it might mess the whole thing up and turn the script into treasure hunt for the root of all errors.
# Feel free to ask questions, come up with ideas or anything
#
# If you somehow manage to turn this failure into a genuine predictor, please make sure to tell me it was helpful to you

import copy
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import requests
from hmmlearn.hmm import *
from scipy.stats import norm

warnings.filterwarnings("ignore")

# Choose the exchange, trading pair, timeframe, history length and aggregation period
def set_parameters(e = 'Binance', c = 'ETH', p = 'minute', l = '200', a = '15'):
    global exch
    exch = e
    global currency
    currency = c
    global period
    period = p
    global limit
    limit = l
    global aggr
    aggr = a

# Download coin histories and create price tables
# Also downloads average BTC/USD price at the time
def prepare_hist_for_training(c):

    def save_hist(path, coin='ETH'):
        listed_history = pull_coin_history(coin)
        df_history = pd.DataFrame(data=listed_history,
                                  columns=['Open', 'Close', 'High', 'Low', 'Volume', 'Time', 'BTC'])

        df_history.to_csv(path, sep='\t', quoting=False, index=False)
        return ()

    def pull_coin_history(coin='ETH'):
        print('Pulling history for %s' % coin)
        call = 'https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym={}&limit={}&aggregate={}&e={}'.format(
            period, coin, currency, limit, aggr, exch)
        attempts = 0
        while attempts < 4:
            try:
                r = json.loads(requests.get(call).text)
                attempts = 5
            except (requests.ConnectionError, ValueError):
                attempts += 1
                time.sleep(3)
                r = None
        if r is None:
            return (None)
        steps = []
        for el in r['Data']:
            steps.append([el['open'], el['close'],
                          el['high'], el['low'],
                          el['volumeto'], el['time']])
        steps = [j for j in steps if not (all([i == 0 for i in j]))][:-1]

        btc_req = 'https://min-api.cryptocompare.com/data/histo{}?fsym=BTC&tsym=USD&limit={}&aggregate={}&e=CCCAGG'.format(period, limit, aggr)
        btc_prices = json.loads(requests.get(btc_req).text)['Data']

        btc_prices = {x['time']:x['close'] for x in btc_prices}

        steps = [x + [btc_prices[x[5]]]for x in steps]

        return (steps)

    def update_hist(filename, coin='ETH'):

        frame = {'day': 86400, 'hour': 3600, 'minute': 60}[period] * int(aggr)


        old_data = pd.read_table(filename, sep='\t', header=0, index_col=None)
        old_data.columns = ['Open', 'Close', 'High', 'Low', 'Volume', 'Time', 'BTC']
        try:
            last_update = old_data.loc[old_data.index[-1], 'Time']
        except:
            print(filename)
            return ()
        now = time.time()
        frames_missing = int(now - last_update) / frame

        if frames_missing != 0:
            temp = pull_coin_history(coin=coin)
            if temp is None:
                save_hist(path=filename, coin=coin)
            else:
                new_days = pd.DataFrame(data=temp[1:],
                                        columns=['Open', 'Close', 'High', 'Low', 'Volume', 'Time', 'BTC'])

                new_days = old_data.loc[:(old_data.shape[0] - 1), :].append(new_days, ignore_index=True)
                new_days = new_days.iloc[(new_days.shape[0] - old_data.shape[0]):, :(new_days.shape[0] - 1)]

                new_days.to_csv(filename, sep='\t', quoting=False, index=False)
                print('Updated history for %s' % coin)

    # Create a folder for history with specified settings if there is none
    directory = './histo_{}_{}_{}_{}_{}/'.format(exch, currency, period, limit, aggr)
    filename = directory + '{}.txt'.format(c)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Download coin history
    # Or take preloaded history
    if not os.path.exists(filename):
        save_hist(filename, coin=c)
    else:
        update_hist(filename, coin=c)
    msg = 'Data for %s loaded' % c

    return(msg)

# Hand holds trained HMMs for coins
# It can store models for multiple coins or just -- whichever you prefer
class Hand:

    def __init__(self):
        self.pos = {}

    def _add_pos(self, pos):
        self.pos[pos.name] = pos

    def _close_pos(self, coin):
        del self.pos[coin]

    # Train HMM for a coin given coin name and a corresponding pandas price df
    def train_coin(self, c, loaded_df):

        # Turns absolute market prices into changes relative to the last price
        def get_frac_data(market):
            frac_data = []
            for i in range(len(market))[1:]:
                prev = market[i - 1]
                now = market[i]
                if ((prev[4] != 0) & (now[0] != 0)):
                    # Now and prev contain information in thee following order
                    # open/close/high/low/volume/btc
                    # The sublists of frac_data contain changes in the following order
                    # dClose, dHigh, -dLow, dVolume, dBTC
                    frac_data.append(
                        [(now[1] - now[0]) / prev[1],
                         (now[2] - now[0]) / prev[1],
                         (now[0] - now[3]) / prev[1],
                         (now[4] - prev[4]) / prev[4],
                         (now[5] - prev[5]) / prev[5]]
                    )
            return (frac_data)

        # Open positions for every coin
        if not (c in self.pos):
            self._add_pos(Position(c))

        hist = loaded_df.values.tolist()

        # Reformat absolute data into a nested list of relative changes
        obs = get_frac_data(hist)

        # Train the HMM
        self.pos[c].get_hmm_model(obs)

        # Some errors may occur
        # Probably should handle them somehow, but meh

        if self.pos[c].hmm is None:
            print('No reliable model for %s\n' % c)
            self._close_pos(c)
            return('err', obs)

        # Estimate price movement by weighing all possible outcomes over all HMM states
        self.pos[c]._get_expected_values(obs)
        if self.pos[c].expected is None:
            print('Can\'t calculate expected values for %s\n' % c)
            self._close_pos(c)
            return ('err', obs)
        # Ignore this (1,1)
        # In another script the return values were somehow important, but not here
        return(1,1)

# Position holds HMM for one coin
# Basically you can replace Hand with Position completely,
# if you predict movement one coin at a time
class Position:

    def __init__(self, name):

        self.buy_price = 0
        self.amount = 0
        self.name = name
        self.hmm = None
        self.expected = {'close':None, 'low':None, 'high':None}
        self.ML = {'close' : None}

    # obs is a list of lists of relative changes
    def get_hmm_model(self, obs):

        def _train_hmm(n_states, obs, iters=200):

            # The features are: dClose, dHigh, -dLow, dVolume, dBTC
            n_features = 5

            # Create an empty model, where each state pulls a value from a Normal distribution
            model = GaussianHMM(n_components=n_states, covariance_type="diag", algorithm='map', n_iter=iters)

            # Equal starting probabilities assumed
            model.startprob_ = np.array([1. / n_states] * n_states)

            # Here I assume 8 possible states, but you can change that too
            # Initially I tried testing 4-6 states models with Normal means
            # defined as sampling means and equal over all state, but I failed.
            #
            # So I tried hardcoding 8 different states, which should correspond to:
            #
            # Steady growth, growing volume,
            # Steady growth, falling volume
            # Fast growth;
            # Steady decline, growing volume;
            # Steady decline, falling volume;
            # Crash decline;
            # Sideways growing volume;
            # Sideways falling volume
            #
            # But that was not anyhow better than equal means models,
            # except for being faster as I didn't need to see which # of states performed better
            # Feel free to tweak these parameters.
            # Perhaps you might actually find the ones that make this predictor work


            # Rows - from, Cols - to
            # If you tweak this one, make sure lines sum up to exactly 1
            # as the probability to leave a state for any other state has to be 1
            model.transmat_ = np.array([
                                        [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05],
                                        [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05],
                                        [0.15, 0.15, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
                                        [0.15, 0.15, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05],
                                        [0.12, 0.12, 0.12, 0.12, 0.21, 0.21, 0.05, 0.05],
                                        [0.12, 0.12, 0.12, 0.12, 0.21, 0.21, 0.05, 0.05],
                                        [0.05, 0.1, 0.1, 0.05, 0.15, 0.15, 0.1, 0.3],
                                        [0.15, 0.15, 0.05, 0.05, 0.2, 0.2, 0.15, 0.1]
                                        ])

            # Rows - states, Cols - features
            model.means_ = np.array([[0.02,0.03, 0.,0.03, 0.03],
                                    [0.02, 0.03, 0., -0.03, 0.03],
                                    [-0.02, 0.0, -0.03, 0.03, 0.03],
                                    [-0.02, 0.0, -0.03, -0.03, 0.03],
                                    [0., 0., 0., -0.01, 0.03],
                                    [0., 0., 0., 0.01, 0.03],
                                    [0.1, 0.12, 0., 0.5, 0.03],
                                    [-0.1, 0., -0.12, 0.5, 0.03]])

            # Transform nested list of relative changes form into numpy matrix form
            obs_stacked = np.column_stack([[i[j] for i in obs] for j in range(n_features)])

            try:
                model = model.fit(obs_stacked)
                # print(list(model.predict(obs_stacked)))
            except ValueError:
                return(None)
            return (model)

        self.hmm = _train_hmm(8, obs)

    # Get price predictions based on:
    # expected value;
    # maximum likelihood
    def _get_expected_values(self,obs):

        def _probabilistic_prediction(obs):

            # Integrate probability at 5% increments
            # Ignore any loss/gain bigger than 50%
            slices = {x / 1000.: None for x in range(-500, 500, 25)}
            n_states = self.hmm.n_components
            obs_stacked = np.column_stack([[i[j] for i in obs] for j in range(len(obs[0]))])
            try:
                last_step_state_probs = self.hmm.score_samples(obs_stacked)[1][-1]
            except:
                return(None, None)

            for s in slices:
                # Introduce expected values for close/low/high fraction change
                probs = {'close': [], 'low': [], 'high': []}
                for j in range(len(last_step_state_probs)):
                    last_state_prob = last_step_state_probs[j]
                    # Transmat: Rows - from, Cols - to
                    trans = self.hmm.transmat_[j]

                    # Calculate sliced probabilities for metrics
                    feats = {0:'close', 1:'high', 2:'low'}
                    for k in feats:
                        mu = self.hmm.means_[:, k]
                        sigma = self.hmm._covars_[:, k]
                        temp = [(norm.cdf(s + 0.025, loc=mu[i], scale=sigma[i]) -
                                    norm.cdf(s, loc=mu[i], scale=sigma[i]))
                                   for i in range(n_states)]
                        # P(last_step = j)*SUM[P(from j to i)*P(s<feat<s+0.05)]over_i = P(s<feat<s+0.025|last_step = j)
                        probs[feats[k]].append(last_state_prob * sum(temp*trans))

                # P(s<feat<s+0.025) = SUM[P(s<feat<s+0.025)]over_j
                slices[s] = {k:sum(probs[k]) for k in probs}
            # expected = {k:sum([(s + 0.025) * slices[s][k] for s in slices]) for k in probs}
            expected = {k: sum([s * slices[s][k] for s in slices]) for k in probs}

            temp = {s:slices[s]['close'] for s in slices}
            ML = {'close': max(temp.items(), key=lambda x: x[1])[0]}
            return (expected, ML)

        # While expected metric estimates all features (dClose, dHigh, dLow, dVolume, dBTC),
        # ML one does so only for dClose
        # That is cos I added ML metric later and didn't bother adding it properly
        temp = _probabilistic_prediction(obs)
        self.expected = temp[0]
        self.ML = temp[1]

def backtest_coin(df, coin, hmm_length):

    temp = []
    temp_ML = []

    # Feel free to test from whatever index, given it will provide enough history for your models to train
    # I have chosen to miss first 1000 lines so that computations don't take too long
    for i in df.index[1001:]:

        train = df.iloc[(i-hmm_length-1):i,]
        new_hand = Hand()
        indic = new_hand.train_coin(coin, train)

        # If the training somehow fails, just take the previous model
        if indic[0] != 'err':
            hand = new_hand
        # Hm...
        # If only I could remember what this is here for?
        elif i!=df.index[1001]:
            hand.pos[c]._get_expected_values(indic[1])
        else:
            hand  = new_hand

        # The next candle's closing price is supposed to be
        # next_open*dClose
        # but I somehow believe it is better to make it
        # last_close*dClose
        change = (df.loc[i,'Close'] - float(train.tail(1)['Close'])) / float(train.tail(1)['Close'])

        try:
            t = df.loc[i,'Time']
            temp.append([t, hand.pos[coin].expected['close'], change])
            temp_ML.append([t, hand.pos[coin].ML['close'], change])

        except KeyError:
            temp.append([t, 'ERROR', change])
            temp_ML.append([t, 'ERROR', change])

    expectancies = (pd.DataFrame(data=temp, columns=['Time', 'Exp', 'Change']),
                    pd.DataFrame(data=temp_ML, columns=['Time', 'Exp', 'Change']))

    return(expectancies)

# So, you get a predicted coin price.
# What is the good enough dClose to buy it?
# You look at which dClose were good enough in the candles
# the training was carried out on and choose as your cutoff.
# Bigger than best cutoff -- you buy, less -- you sell
#
# A major problem I noticed here is all cutoffs are lop-sided
# Somehow, traininng data suggests that your maximal gain is achieved
# if you only sell or only buy during all trading candles
# It is either the indication of poor model choice (which it could be)
# or poor implementation (which is also likely)
# I wanted to test HMM approach in MatLab to see if it will make any difference.
# If it performed just as badly, I would assume the whole HMM approach is not good for crypto.
# But I don't have neither the time nor desire to rework this effort wasting pit of a code
# into another language.
def cutoff_predictions(df, hmm_len):

    try:
        df[df['Exp'] == 'ERROR', 'Exp'] = 0.0
        df = df.apply(pd.to_numeric)
    except:
        pass

    df['Gain'] = 0.0
    df['Signal'] = 'hold'
    df['Cutoff'] = 0.0

    for i in range(hmm_len, df.shape[0]):

        print(i)
        try:
            if df.loc[i-1, 'Exp'] < best_thr:
                new_gain = -1*df.loc[i-1, 'Change']
            else:
                new_gain = df.loc[i-1, 'Change']
            if new_gain > 0:
                if df.loc[i, 'Exp'] < best_thr:
                    df.loc[i, 'Gain'] = -1 * df.loc[i, 'Change']
                    df.loc[i, 'Signal'] = 'sell'
                else:
                    df.loc[i, 'Gain'] = df.loc[i, 'Change']
                    df.loc[i, 'Signal'] = 'buy'
                df.loc[i, 'Cutoff'] = best_thr
                continue
            else:
                new = round(df.loc[i-1, 'Exp'],3)
                passed = round(df.loc[i - hmm_len - 1, 'Exp'], 3)
                thr = [best_thr, new+0.001, new-0.001, passed + 0.001, passed - 0.001]
                thr.sort()
                part_df = copy.deepcopy(df.iloc[(i-hmm_len):i, : ])
        except:
            part_df = copy.deepcopy(df.iloc[(i-hmm_len):i, : ])
            thr = list(set([round(j,3) for j in part_df['Change']]))
            thr.sort()
            t_prev = thr[0]
            thr = thr[1:-1]

        # thr = [-.017]
        best_gain = -100.

        part_df['Signal'] = 'buy'
        part_df['Gain'] = part_df['Change']

        for t in thr:

            smaller = list(set(part_df.loc[(part_df['Exp'] < t)&(part_df['Exp'] >= t_prev)].index))

            part_df.loc[smaller, 'Gain'] = -1*part_df['Change']
            part_df.loc[smaller, 'Signal'] = 'sell'
            gains = part_df['Gain'].sum()

            t_prev = t
            if gains > best_gain:
                best_thr = t

        t = best_thr
        df.loc[i, 'Cutoff'] = t
        if df.loc[i, 'Exp'] < t:
            df.loc[i, 'Gain'] = -1 * df.loc[i, 'Change']
            df.loc[i, 'Signal'] = 'sell'
        else:
            df.loc[i, 'Gain'] = df.loc[i, 'Change']
            df.loc[i, 'Signal'] = 'buy'
        df.loc[i, 'Cutoff'] = best_thr

    return(df)

# Basically see how well you perform
# The strategy is buy when sell signal is followed by a buy one
# and sell when it is the other way round
# Sure, you can assume that you either buy or sell each candle,
# but it turns out to be even more devastating
def overall_edge(c, hist_folder):

    hist_folder = hist_folder.strip('/')
    prices = pd.read_table('{}/{}.txt'.format(hist_folder,c))
    actions = pd.read_table('./{}_decis.txt'.format(c))
    actions = actions[actions['Signal']!='hold']

    prices = prices[prices['Time'] >= actions.loc[actions.index[0],'Time']]
    prices['Signal'] = 'hold'
    prices['Signal'] = list(actions['Signal'])

    gains = []
    last_sell = None
    last_buy = None

    for i in prices.index:

        if i == prices.index[0]:
            if prices.loc[i,'Signal'] == 'buy':
                last_buy = prices.loc[i,'Open']
            else:
                last_sell = prices.loc[i, 'Open']

            if len(set(prices['Signal'])) == 1:
                return('ERROR')
            continue


        if prices.loc[i, 'Signal'] != prices.loc[i-1, 'Signal']:
            # sell
            # ...
            # sell
            # buy
            if prices.loc[i, 'Signal'] == 'buy':
                gains.append((last_sell - prices.loc[i, 'Open'])/(last_sell))
                last_sell = None
                last_buy = prices.loc[i, 'Open']
            # buy
            # ...
            # buy
            # sell
            if prices.loc[i, 'Signal'] == 'sell':
                gains.append((prices.loc[i, 'Open']-last_buy) / (last_buy))
                last_sell = prices.loc[i, 'Open']
                last_buy = None

    xs = [1+x for x in gains]
    overall = np.prod(xs)
    return(overall)


coins = ['FUN', 'MANA', 'MAID', 'WAVES', 'GRC']
set_parameters(e = 'Bittrex', c = 'BTC', p = 'hour', l = '2000', a = '1')

for c in coins:
    # Uncomment if you run this for the first time
    # prepare_hist_for_training(c)
    df = pd.read_table('./histo_Bittrex_BTC_hour_2000_1/{}.txt'.format(c))
    x = backtest_coin(df, c, 200)[0]
    x.to_csv('%s_exp.txt' % c, sep='\t', index=None)
    y = cutoff_predictions_one(x)
    print(c)
    y.to_csv('%s_decis.txt' % c, sep='\t', index=None)
    print('{} edge is {}%'.format(c, round(x['Gain'].mean() * 100, 3)))


coins = ['FUN', 'MANA', 'MAID', 'WAVES', 'GRC']
hist_folder = './histo_Bittrex_BTC_hour_2000_1/'
for c in coins:
    print(c)
    temp = overall_edge(c, hist_folder)
    if temp != 'ERROR':
        print('{}% edge'.format(round(temp*100,2)))
    else:
        print(temp)

# Final note
# What might actually kickstart this predictor:
# aggregating more data for coins to include periods where all the mentioned states are present
# (no wonder the HMM can't determine a price spike, when it was trained in a downtrend);
# adding another dimension or two to the transition matrix
# (so that the transition is dependent on 2 previous candles and not just one. I am not sure the
# package I am using is capable of this);
#
# Combining signals from small and big candles didn't work for me
# and just made the whole thing work longer. So that is definetely not the first
# optimisation to try if you wondered.

# Fedor Galkin
# lotus28@mail.ru