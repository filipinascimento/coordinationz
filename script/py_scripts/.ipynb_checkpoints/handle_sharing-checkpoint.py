import pandas as pd
import sys
import argparse
sys.path.insert(0, 
                '/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/script/helper')
from helper import *



def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Create synchronizations coordination network')
    
    parser.add_argument('--input',
                        dest='input_path',
                        help='Input file path')
    
    parser.add_argument('--output',
                        dest='output_path',
                        help='Output file path')
    
    parser.add_argument('--campaign-name',
                        dest='campaign_name',
                        help='Name of campaign')
    
    # parser.add_argument('--user-file',
    #                     dest='user_filename',
    #                     help='User file name')
    
    parser.add_argument('--tweet-file',
                        dest='user_filename',
                        help='User file name')
                        
    return parser.parse_args()


def handle_change_mentions(df, campaign, reply_only=True, 
                           ops=True, save=True, path=None):
    '''
    Tests whether there is handle sharing cases in data by comparing 
    mentioned username and meta data of mentioned user
    
    :param df: dataframe
    :param campaign: name of campaign
    :param reply_only: boolean test whether to test only on replies
    :param ops: boolean to set whether it is a operation data 
    for the control data
    '''
    if reply_only == False:
        return
    
    df = df.loc[df['is_retweet'] == False]

    df = extract_mentions(df)
    
    df = df.astype({'userid': int})
    
    
    
    df_mentions = df.loc[~df['user_mentions'].isnull()]
    df_mentions = df_mentions.reset_index(drop=True)
    df_mentions = df_mentions[['tweetid', 
                               'userid', 
                               'user_screen_name',
                               'in_reply_to_userid', 
                               'tweet_text', 
                               'user_mentions',
                               'tweet_time',
                               'mentions_from_text']]
    all_names = []
    
    if ops == True:
        type_of = 'ops'
        #list of userids
        df_mentions['user_mentions'] = df_mentions['user_mentions'].apply(
            lambda x: x[1:-1].split(',')
        )
    else:
        df_mentions['user_mentions'] = df_mentions['user_mentions'].apply(
            lambda x: [ ids['id'] for ids in x]
        )
        type_of = 'control'
        
    for index, i in df_mentions['user_mentions'].iteritems():
        row = df_mentions.iloc[index]
        reply_time = row['tweet_time']
        l1 = row['mentions_from_text']
        
        if len(l1) == 0 or len(i) == 0:
            continue
        
        if len(l1) > 1:
            continue
            
        l2 = i[0]
        
        # l3 = list(product(l1, l2))
        
        #l2 is mentioned userid from data, l1 is mentioned username
        l3 = (l1[0], l2)
        
        #m is the mentioned userid
        for (mentioned_name, m) in [l3]:
            if m is None or len(m) == 0:
                continue
                
            m = int(m)
            df_name = df.loc[(df['userid'] == m)]
            
            if len(df_name) == 0:
                continue

            #screen name of replied user
            names = df_name['user_screen_name'].unique()
            
            #mention extracted from text, mentioned_name
            # mentions = df_mentions.iloc[index]['mentions_from_text']
            names = '@' + names[0]

            #user_screen_name, mentions, userid, campaign, reply_time
            all_names.append([names, mentioned_name, m, 
                              campaign, reply_time])
            print(all_names)
            
        if save == True:
            
            print(len(all_names))
            df_handle = pd.DataFrame(all_names, 
                         columns=['username', 'mentioned_name', 
                                  'userid', 'campaign', 'reply_time'])
            df_handle = df_handle.drop_duplicates()
            
            # print(df.loc[df['userid'] == 1268645927532867584]['user_screen_name'].unique())
            # print(df.loc[df['userid'] == 1213475439802347520]['user_screen_name'].unique())
            # print(df.loc[df['user_screen_name'] == 'Prettyglosh2']['userid'].unique())
           
            df_handle.to_pickle(
                f'{path}/{campaign}_handle_change_{type_of}.pkl.gz')
            return df_handle
        
    return all_names


def calculate_similarity_in_names(df, filename_with_path):
    if len(df) == 0:
        return
    print(' --------- here --------- ')
    
    df['username'] = df['username'].apply(lambda x: x.translate(
        str.maketrans('', '', string.punctuation)))
    
    df['mentioned_name'] = df['mentioned_name'].apply(lambda x: x.translate(
        str.maketrans('', '', string.punctuation)))
    
    df['username'] = df['username'].apply(lambda x: x.strip())
    df['mentioned_name'] = df['mentioned_name'].apply(lambda x: x.strip())
    df['username'] = df['username'].str.rstrip('.')
    df['mentioned_name'] = df['mentioned_name'].str.rstrip('.')
    df['mentioned_name'] = df['mentioned_name'].str.lower()
    df['username'] = df['username'].str.lower()
    
    df['username'] = df['username'].apply(
        lambda x: remove_emoji(x))
    df['mentioned_name'] = df['mentioned_name'].apply(
        lambda x: remove_emoji(x))
    
    df = df.drop_duplicates()
    
    #Checking the similarity in userscreen names
    df_diff = df.loc[df['username'] != df['mentioned_name']]
    
    df_diff = similarity_in_display_name(df_diff, 
                               fields=['username', 'mentioned_name'],
                               include_in_df=True)
    df_diff.to_pickle(
            f'{filename_with_path}')

    return df_diff
    
    
def test_if_multiple_userid(df):
    #Checking if the mentioned name have been used twice
    df_new = pd.DataFrame()
    
    df_new = df_new.append(df[['username', 'userid']], 
                           ignore_index=True)
    df_new.rename(columns={'username': 'mentioned_name'}, 
                           inplace=True)
    df_new = df_new.append(df[['mentioned_name', 'userid']], 
                           ignore_index=True)
    
    df_new = df_new.drop_duplicates()
    
    df_merge = df_new.merge(df_new, on=['mentioned_name'])
    
    df_merge = df_merge.drop_duplicates()

    df_merge = df_merge.loc[df_merge['userid_x'] != df_merge['userid_y']]
    
    print(df_merge)


#This does not work because one can mention the user in reply,
#replied user mention is not exclusively included in tweet

def handle_sharing(df):
    '''
    Tests whether there is any handle sharing cases by testing
    repeated userid or repeated userscreen name
    
    :param df: dataframe
    '''
    
    df = df.drop_duplicates()
    
    #@screen name
    df_username = (df
                .groupby(['user_screen_name'])['userid']
                .apply(lambda x: list(np.unique(x)))
                .to_frame('userid')
                .reset_index())
    df_username['count_userid'] = df_username['userid'].map(len)
    
    print('Count of users with same screen name',
      len(df_username.loc[df_username['count_userid'] > 1]))
        
    #groupby userid
    df_username = (df
                .groupby(['userid'])['user_screen_name']
                .apply(lambda x: list(np.unique(x)))
                .to_frame('user_screen_name')
                .reset_index())
    
    df_username['count_names'] = df_username['user_screen_name'].map(len)
    
    print('Number of users with more than one screen name',
          len(df_username.loc[df_username['count_names'] > 1]))


def handle_sharing_test(df, args):
    '''
    Tests whether there are any handle sharing cases or not
    
    :param df: User dataframe
    :param args: Arguments passed through command
    '''
  
    print('\n----- Start: Handle sharings test ---------')
    handle_sharing(df)
    
    campaign_name = args.campaign_name
    
    df = handle_change_mentions(df, campaign_name, 
                        reply_only=True, 
                        ops=True, save=True, 
                        path=args.output_path)
    
    filename = f'{campaign_name}_handle_change_cleaned_ops.pkl.gz'
    filename_with_path = os.path.join(args.output_path, filename)
    
    print(df)
    
    if len(df) != 0:
        # df = calculate_similarity_in_names(df, filename_with_path)
    
        test_if_multiple_userid(df)
    
    print('----- End: Handle sharings test ---------\n')


    
def main():
    try:
        print('----- Start: Parsing the arguments ---------')
        args = parse_args()
        print('----- End: Parsing the arguments -----------\n')
        
        print('----- Start: Reading User data ---------')
        df = read_data(args.input_path, args.user_filename)
        print('----- End: Reading Tweet data ---------\n')
        
        
        print('----- Start: Handle sharing test ---------')
        handle_sharing_test(df, args)
        print('----- End: Handle sharing test ---------\n')
        
    except Exception as e:
        print(e)
        return e
    
if __name__ == "__main__":
    main()
    
    
    
# python handle_sharing.py --input=/N/slate/potem/data/derived/all_tweets/2021_12/uganda_0621 --output=/N/slate/potem/data/derived/combined/handle_change --campaign-name=uganda_0621 --tweet-file=uganda_0621_tweets.pkl.gz
                    
