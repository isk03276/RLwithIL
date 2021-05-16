from common.rl_utils import RLUtils

def check_divde_list():
    data1 = [False, False,False, False,False, False,True]
    data2 = [False, False, True, True, False, False]

    case1 = RLUtils.divice_list(data1, 3)
    case2 = RLUtils.divice_list(data2, 3)

    assert case1 == [2, 5, 6]
    assert case2 == [2, 3, 5]

def check_td_nstep_return():
    dones1 = [False, False, False, False, False, False,True]
    dones2 = [False, False, True, False, False, False]

    rews1 = [0, 0, 0, 0, 0, 0, 1]
    rews2 = [0.3, 0, 1, 0, 0, 1]

    values1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    values2 = [0.1, 0.3, 1, 0.1, 0.4, 1.]

    gamma = 0.9

    #case1 = RLUtils.get_nstep_td_return(rews1, values1, dones1, gamma, 3)
    case2 = RLUtils.get_nstep_td_return(rews2, values2, dones2, gamma, 3)

    #print(case1)
    print(case2)




if __name__ == "__main__":
    check_divde_list()