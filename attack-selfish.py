"""
bitcoin network simulator - btcsim
Copyright (C) 2013 Rafael Brune <mail@rbrune.de>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 and
only version 2 as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
02110-1301, USA.
"""

import os
import sys
import numpy
import pandas as pd
from pandas import DataFrame, Series
import pylab
import xlsxwriter
from heapq import *

from btcsim import *

def selfish_miners_simulation(alpha,total_miners, num_selfish,latency = [0.02,0.2], bandwidth = [10*1024,200*1024], rand_honest_mine_hash = False):
    class BadMiner(Miner):
        chain_head_others = '*'
        privateBranchLen = 0
        
        def add_block(self, t_block):
            self.blocks[hash(t_block)] = t_block
            if (self.chain_head == '*'):
                self.chain_head = hash(t_block)
                self.chain_head_others = hash(t_block)
                self.mine_block()
                return
            
            if (t_block.miner_id == self.miner_id) and (t_block.height > self.blocks[self.chain_head].height):
                delta_prev = self.blocks[self.chain_head].height - self.blocks[self.chain_head_others].height
                self.chain_head = hash(t_block)
                self.privateBranchLen += 1
                if (delta_prev == 0) and (self.privateBranchLen == 2):
                    self.announce_block(self.chain_head)
                    self.privateBranchLen = 0
                self.mine_block()
            
            if (t_block.miner_id != self.miner_id) and (t_block.height > self.blocks[self.chain_head_others].height):
                delta_prev = self.blocks[self.chain_head].height - self.blocks[self.chain_head_others].height
                self.chain_head_others = hash(t_block)
                if delta_prev <= 0:
                    self.chain_head = hash(t_block)
                    self.privateBranchLen = 0
                elif delta_prev == 1:
                    self.announce_block(self.chain_head)
                elif delta_prev == 2:
                    self.announce_block(self.chain_head)
                    self.privateBranchLen = 0
                else:
                    iter_hash = self.chain_head
                    # the temp is in case we get too far ahead (in case we have >51%)
                    temp = 0
                    if delta_prev >= 6: temp = 1
                    while self.blocks[iter_hash].height != t_block.height + temp:
                        iter_hash = self.blocks[iter_hash].prev
                    self.announce_block(iter_hash)
                self.mine_block()
    

    t = 0.0
    event_q = []

    # root block
    seed_block = Block(None, 0, t, -1, 0, 1)


    # set up some miners with random hashrate
    total_miners = total_miners
    num_selfish = num_selfish
    num_honest = total_miners - num_selfish
    hash_total_selfish = alpha

    if rand_honest_mine_hash:
        hashrates = numpy.random.exponential(1.0, num_honest)
        hashrates = (hashrates/hashrates.sum()) * (1 - hash_total_selfish)
    else:
        hashrates = numpy.asarray([i for i in range(int(num_honest))])
        hashrates = (hashrates/hashrates.sum()) * (1 - hash_total_selfish)


    hashrates = numpy.append(hashrates, [hash_total_selfish/num_selfish for i in range(num_selfish)])
#    hashrates = [ 0.08324844,  0.08552132,  0.11136445,  0.20999066,  0.10177589,
#     0.00809923,  0.1       ,  0.1       ,  0.1       ,  0.1       ]


#    print('##############')
#    print(hashrates)
#    print('##############')

    miners = []
    for i in range(total_miners):
        if i < num_honest:
            miners.append(Miner(i, hashrates[i] * 1.0/600.0, 200*1024, 1024*200*numpy.random.random(), seed_block, event_q, t))
        else:
            miners.append(BadMiner(i, hashrates[i] * 1.0/600.0, 200*1024, 1024*200*numpy.random.random(), seed_block, event_q, t))


    # make the strong miner a bad miner
# print(i)




    # add some random links to each miner

    for i in range(total_miners):
        for k in range(4):
            j = numpy.random.randint(0, total_miners)
            if i != j:
                latency_val = latency[0] + latency[1]*numpy.random.random()
                bandwidth_val = bandwidth[0] + bandwidth[1]*numpy.random.random()

                miners[i].add_link(j, latency_val, bandwidth_val)
                miners[j].add_link(i, latency_val, bandwidth_val)


    # simulate some days of block generation
    curday = 0
    maxdays = 3*7*24*60*60
    while t < maxdays:
        t, t_event = heappop(event_q)
        #print('%08.3f: %02d->%02d %s' % (t, t_event.orig, t_event.dest, t_event.action), t_event.payload)
        miners[t_event.dest].receive_event(t, t_event)
        
        if t/(24*60*60) > curday:
            print('day %03d' % curday)
            curday = int(t/(24*60*60))+1




    ##
    mine = miners[0]
    t_hash = mine.chain_head

    rewardsum = 0.0
    for i in range(total_miners):
        miners[i].reward = 0.0

    main_chain = dict()
    main_chain[hash(seed_block)] = 1

    while t_hash != None:
        t_block = mine.blocks[t_hash]
        
        if t_hash not in main_chain:
            main_chain[t_hash] = 1
        
        miners[t_block.miner_id].reward += 1
        rewardsum += 1
        
        if False:
            pylab.plot([mine.blocks[t_block.prev].time, t_block.time], [mine.blocks[t_block.prev].height, t_block.height], cols[t_block.miner_id%4])

        t_hash = t_block.prev



    orphans_set = set()
    orphans = 0
    for i in range(total_miners):
        for t_hash, block in miners[i].blocks.items():
            if t_hash not in main_chain:
                orphans += 1
                orphans_set.add((t_hash, block.miner_id))
            # draws the chains
            if miners[i].blocks[t_hash].height > 1:
                cur_b = miners[i].blocks[t_hash]
                pre_b = miners[i].blocks[cur_b.prev]
#                pylab.plot([hashrates[pre_b.miner_id], hashrates[cur_b.miner_id]], [pre_b.height, cur_b.height], 'k-')
    orphans = len(orphans_set)

    return(seed_block, hash, miners, main_chain, hashrates, orphans, rewardsum)



# data analysis

#pylab.figure()
#
#cols = ['r-', 'g-', 'b-', 'y-']
#
#mine = miners[0]
#t_hash = mine.chain_head
#
#rewardsum = 0.0
#for i in range(total_miners):
#    miners[i].reward = 0.0
#
#main_chain = dict()
#main_chain[hash(seed_block)] = 1
#
#while t_hash != None:
#    t_block = mine.blocks[t_hash]
#
#    if t_hash not in main_chain:
#        main_chain[t_hash] = 1
#
#    miners[t_block.miner_id].reward += 1
#    rewardsum += 1
#
#    if t_block.prev != None:
#        pylab.plot([mine.blocks[t_block.prev].time, t_block.time], [mine.blocks[t_block.prev].height, t_block.height], cols[t_block.miner_id%4])
#
#    t_hash = t_block.prev
#
#pylab.xlabel('time in s')
#pylab.ylabel('block height')
#pylab.draw()
#
#pylab.figure()
#
#pylab.plot([0, numpy.max(hashrates)*1.05], [0, numpy.max(hashrates)*1.05], '-', color='0.4')
#
#for i in range(total_miners):
#    #print('%2d: %0.3f -> %0.3f' % (i, hashrates[i], miners[i].reward/rewardsum))
#    if i < num_honest:
#        pylab.plot(hashrates[i], miners[i].reward/rewardsum, 'k.')
#    else:
#        pylab.plot(hashrates[i], miners[i].reward/rewardsum, 'rx')
#
#pylab.xlabel('hashrate')
#pylab.ylabel('reward')
#
#
#
#pylab.figure()
#orphans = 0
#for i in range(total_miners):
#    for t_hash in miners[i].blocks:
#        if t_hash not in main_chain:
#            orphans += 1
#        # draws the chains
#        if miners[i].blocks[t_hash].height > 1:
#            cur_b = miners[i].blocks[t_hash]
#            pre_b = miners[i].blocks[cur_b.prev]
#            pylab.plot([hashrates[pre_b.miner_id], hashrates[cur_b.miner_id]], [pre_b.height, cur_b.height], 'k-')

#pylab.ylabel('block height')
#pylab.xlabel('hashrate')
#pylab.ylim([0, 100])
#
#print('Orphaned blocks: %d (%0.3f)' % (orphans, orphans/mine.blocks[mine.chain_head].height))
#print('##########MINER_BLOCKS############')
#print(sum([len(miners[i].blocks) for i in range(len(miners))]))
#print('##########MINER_BLOCKS############')
#print('##########mainchain############')
#print(len(main_chain))
#print('##########mainchain############')
#print('Average block height time: %0.3f min' % (mine.blocks[mine.chain_head].time/(60*mine.blocks[mine.chain_head].height)))
##print(total_mined_blocks())
#
#
#
#
#pylab.draw()
#pylab.show()


# setting parameter lists
alphas = [0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1]
nums_selfish = [1, 2, 3, 4 ,5, 10, 25, 50, 75, 100]
latency = [0.020,0.2]
bandwidth = [10*1024,200*1024]

# Column labels are simply alpha and num_selfish values
rows = alphas
cols = nums_selfish

# Instatiate dataframes with appropriate col/row labels
rewards_df = DataFrame()
orphans_df = DataFrame(columns=cols, index=rows)
honest_rewards_total_df = DataFrame(columns=cols, index=rows)
selfish_rewards_total_df = DataFrame(columns=cols, index=rows)
selfish_rewards_sd_df = DataFrame(columns=cols, index=rows)
orphan_rate_df = DataFrame(columns=cols, index=rows)


for alpha in alphas:
    for num_selfish in nums_selfish:
        
        # Set max number of total miners
        total_miners = int(numpy.sqrt(num_selfish) * 15)
        
        # Invoke selfish miner function
        seed_block, hash, miners, main_chain, hashrates, orphans, rewardsum  = selfish_miners_simulation(alpha,total_miners, num_selfish)
        
        # Gather necessary info
        rewards = []
        selfish_rewards = []
        honest_rewards = []
        rewards = [miners[i].reward/rewardsum for i in range(total_miners)]
#        selfish_rewards = [rewards[i] for i in range(num_selfish, total_miners)]
#        honest_rewards = [rewards[i] for i in range(num_selfish)]
        total_blocks_set = set()
        print (sum(selfish_rewards), sum(honest_rewards))
        
        # Get total number of mined blocks
        for i in range(len(miners)):
            for key, val in miners[i].blocks.items():
                total_blocks_set.add((val.miner_id, val.height))
    
        total_blocks_mined = len(total_blocks_set)

        # If you change any of the alpha, num_selfish etc arrays, be sure to change the 150 here and any other hard coded values
        nas = ['NA' for i in range(150 - len(rewards))]
        rewards_full = rewards + nas
        
        # Add to dfs accordingly
        rewards_df[str(alpha) +  '_' + str(num_selfish)] = pd.Series(rewards_full)
        orphans_df.loc[alpha, num_selfish] = orphans
        selfish_rewards_total_df.loc[alpha, num_selfish]  = sum(rewards[total_miners - num_selfish:total_miners])
        honest_rewards_total_df.loc[alpha, num_selfish]  = sum(rewards[:total_miners - num_selfish])
        selfish_rewards_sd_df.loc[alpha, num_selfish]  = numpy.std(rewards[total_miners - num_selfish:total_miners])
        orphan_rate_df.loc[alpha, num_selfish]  = orphans/total_blocks_mined

# Check
        print(total_blocks_mined, orphans)



writer = pd.ExcelWriter('thesis_simulattion_results_F2_latencynorm.xlsx', engine='xlsxwriter')
rewards_df.to_excel(writer, sheet_name = 'all rewards')
orphans_df.to_excel(writer, sheet_name = 'orphans')
honest_rewards_total_df.to_excel(writer, sheet_name = 'honest')
selfish_rewards_total_df.to_excel(writer, sheet_name = 'selfish')
selfish_rewards_sd_df.to_excel(writer, sheet_name = 'std')
orphan_rate_df.to_excel(writer, sheet_name = 'orphan rate')
writer.save()
