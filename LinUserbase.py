from _commons import warn, error, create_dir_path
import numpy as np
import time
from movielens import MovieLens
import random



#'binary_unknown'
class LinUserbase:
    def __init__(self, alpha, dataset=None, max_items=500, allow_selecting_known_arms=True, fixed_rewards=True,
                 prob_reward_p=0.9):
        if dataset is None:
            self.dataset = MovieLens(variant='ml-100k',
                                     pos_rating_threshold=4,
                                     data_augmentation_mode='binary_unknown')
        else:
            self.dataset = dataset
        self.dataset.shrink(max_items)
        self.dataset.add_random_ratings(num_to_each_user=3)
        self.alpha = alpha
        self.fixed_rewards = fixed_rewards
        self.prob_reward_p = prob_reward_p
        self.users_with_unrated_items = np.array(range(self.dataset.num_users))
        self.monitored_user = np.random.choice(self.users_with_unrated_items)
        self.allow_selecting_known_arms = allow_selecting_known_arms
        self.d = self.dataset.arm_feature_dim
        self.b = np.zeros(shape=(self.dataset.num_items, self.d))

        # More efficient way to create array of identity matrices of length num_items
        print("\nInitializing matrix A of shape {} which will require {}MB of memory."
              .format((self.dataset.num_items, self.d, self.d), 8 * self.dataset.num_items * self.d * self.d / 1e6))

        # 这个matrix A有三个维度，分别为被推荐的item的数量，每个矩阵的维度都是arm_feature的维度，是单位矩阵
        self.A = np.repeat(np.identity(self.d, dtype=float)[np.newaxis, :, :], self.dataset.num_items, axis=0)
        print("\nLinUCB successfully initialized.")

    def find_top_k_similarity(self, t, k = 2):
        """
        Choose an arm to pull = item to recommend to user t that he did not rate yet.
        :param t: User_id of user to recommend to.
        :param unknown_item_ids: Indexes of items that user t has not rated yet.
        :return: the top k nearest user of t and its similarity
        """
        R = self.dataset.orig_R_shrink
        user_num = R.shape[0]
        sim = np.zeros(user_num)
        self.k= k

        for user_id in range(user_num):
            if user_id != t:
                simlarity = np.dot(R[t], R[user_id]) / (np.linalg.norm(R[t]) * np.linalg.norm(R[user_id]))
                if np.isnan(simlarity) | (simlarity > 1):
                    sim[user_id] = -1
                else:
                    sim[user_id] = simlarity
            else:
                sim[user_id] = 1

        user_pos_index = np.where(sim > 0)
        user_top_k = np.argsort(sim)[-k:]

        # 两个集合取交集
        user_top_k = np.intersect1d(user_pos_index, user_top_k)
        sim_top_k = sim[user_top_k]

        return user_top_k, sim_top_k

    def choose_arm(self, t, unknown_item_ids, verbosity):
        """
        Choose an arm to pull = item to recommend to user t that he did not rate yet.
        :param t: User_id of user to recommend to.
        :param unknown_item_ids: Indexes of items that user t has not rated yet.
        :return: Received reward for selected item = 1/0 = user liked/disliked item.
        """
        A = self.A
        b = self.b


        top_k_user, top_k_user_sim = self.find_top_k_similarity(t=t)
        top_k_user_weight = top_k_user_sim / np.sum(top_k_user_sim)
        top_k_user_weight = top_k_user_weight.reshape(-1,1) # 将行向量转化为列向量

        count = 0;
        #item_ids = unknown_item_ids
        item_ids = range(self.dataset.num_items)
        P_t = np.zeros(shape=(len(top_k_user), self.dataset.num_items))
        for neighbor in top_k_user:
            arm_features = self.dataset.get_features_of_current_arms(t=neighbor)
            # arm_features 是一个由评分和genre拼合而成的矩阵

            p_t = np.zeros(shape=(arm_features.shape[0],), dtype=float)
            p_t -= 9999  # I never want to select the already rated items

            if self.allow_selecting_known_arms:
                item_ids = range(self.dataset.num_items)
                p_t += 9999

            for a in item_ids:  # iterate over all arms
                '''
                x_ta = arm_features[a].reshape(arm_features[a].shape[0], 1)  # make a column vector

                A_a_inv = np.linalg.inv(A[a])
                theta_a = A_a_inv.dot(b[a])
                p_t[a] = theta_a.T.dot(x_ta) + self.alpha * np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))
                '''

                #if a in self.dataset.get_uknown_items_of_user(neighbor):
                #print(self.dataset.get_uknown_items_of_user(neighbor))
                #print(unknown_item_ids)
                #print('--------------')
                if a in self.dataset.get_uknown_items_of_user(neighbor):
                    # 调取arm_feature矩阵的第a行，第a行是user_id 对所有历史item的评分，再与这个电影的genre拼合起来
                    x_ta = arm_features[a].reshape(arm_features[a].shape[0], 1)  # make a column vector

                    A_a_inv = np.linalg.inv(A[a])
                    theta_a = A_a_inv.dot(b[a])
                    p_t[a] = theta_a.T.dot(x_ta) + self.alpha * np.sqrt(x_ta.T.dot(A_a_inv).dot(x_ta))
                else:
                    p_t[a] = self.dataset.recommend(user_id=neighbor, item_id=a,
                                     fixed_rewards=self.fixed_rewards, prob_reward_p=self.prob_reward_p)


            P_t[count] = p_t
            count += 1

        # 之前P_t是一个行为k个最近user，列为每个item的矩阵，形状为(k,p)
        # similar_weight 经转化后是一个k*1形状的矩阵
        # 现在讲P_t转置，变为(p,k)，与k*1的矩阵相乘，得到p个item的加权均值
        P_t = P_t.T
        p_t = np.matmul(P_t,top_k_user_weight).flatten()#得到了一个p的行向量

        p_t_unknown = p_t[unknown_item_ids];

        max_p_t = np.max(p_t)
        if max_p_t <= 0:
            print("User {} has max p_t={}, p_t={}".format(t, max_p_t, p_t))

        # I want to randomly break ties, np.argmax return the first occurence of maximum.
        # So I will get all occurences of the max and randomly select between them
        max_idxs = np.argwhere(p_t == max_p_t).flatten()
        a_t = np.random.choice(max_idxs)  # idx of article to recommend to user t

        # observed reward = 1/0
        r_t = self.dataset.recommend(user_id=t, item_id=a_t,
                                     fixed_rewards=self.fixed_rewards, prob_reward_p=self.prob_reward_p)

        if verbosity >= 2:
            print("User {} choosing item {} with p_t={} reward {}".format(t, a_t, p_t[a_t], r_t))

        x_t_at = arm_features[a_t].reshape(arm_features[a_t].shape[0], 1)  # make a column vector

        # 对于每一个arm 都会维护一个A maxtrix，只有在这个item被推荐到了，这个item的A矩阵才会有更新
        A[a_t] = A[a_t] + x_t_at.dot(x_t_at.T)
        b[a_t] = b[a_t] + r_t * x_t_at.flatten()  # turn it back into an array because b[a_t] is an array

        return r_t


    # 每一个epoch是将每个pool中的每一个user都推荐一遍
    def run_epoch(self, verbosity=2):
        """
        Call choose_arm() for each user in the dataset.
        :return: Average received reward.
        """
        rewards = []
        start_time = time.time()

        for i in range(self.dataset.num_users):
            start_time_i = time.time()
            user_id = self.dataset.get_next_user()
            #print(user_id)
            # user_id = 1
            unknown_item_ids = self.dataset.get_uknown_items_of_user(user_id)
            if len(unknown_item_ids) == 0:
                continue;

            if self.allow_selecting_known_arms == False:
                if user_id not in self.users_with_unrated_items:
                    continue

                if unknown_item_ids.size == 0:
                    print("User {} has no more unknown ratings, skipping him.".format(user_id))
                    self.users_with_unrated_items = self.users_with_unrated_items[
                        self.users_with_unrated_items != user_id]
                    continue

            rewards.append(self.choose_arm(user_id, unknown_item_ids, verbosity))
            time_i = time.time() - start_time_i
            if verbosity >= 2:
                print("Choosing arm for user {}/{} ended with reward {} in {}s".format(i, self.dataset.num_users,
                                                                                       rewards[i], time_i))

        total_time = time.time() - start_time
        avg_reward = np.average(np.array(rewards))
        return avg_reward, total_time

    def run(self, num_epochs, verbosity=1):
        """
        Runs run_epoch() num_epoch times.
        :param num_epochs: Number of epochs = iterating over all users.
        :return: List of average rewards per epoch.
        """
        self.users_with_unrated_items = np.array(range(self.dataset.num_users))
        avg_rewards = np.zeros(shape=(num_epochs,), dtype=float)
        for i in range(num_epochs):
            avg_rewards[i], total_time = self.run_epoch(verbosity)

            if verbosity >= 1:
                print(
                    "Finished epoch {}/{} with avg reward {} in {}s".format(i, num_epochs, avg_rewards[i], total_time))
        return avg_rewards
