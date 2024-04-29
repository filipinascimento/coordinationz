






def tweet_client_coordination(data, campaign_type, network_save_path,
                              graph_path, threshold=1
                             ):
    df_unique_client, len_uniques = extract_tweet_client(data)

    print('\n----- Start: Create co-client share strings  -----')
    df_string = convert_tweetids_to_string(df_unique_client, 
                                           field='tweet_client_name',
                                           filter_threshold=True,
                                           threshold=threshold
                                          )
    print('\n----- End: Create co-client share strings -----')

    print('\n----- Start: Create bi-partite network -----')
    df_network = create_user_projection(df_unique_client, 
                                        'tweet_client_name',
                                        userid=list(df_string['userid'])
                                       )
    print('\n----- End: Create bi-partite network -----')

    if len(df_network) == 0:
        print('\n --- No data to continue ---- \n')

        return [new_campaign, campaign_type, 0, 0, 0, len_uniques]


    print('\n----- Start: Calculate tfidf vector -----')
    df_string = calculate_tf_idf_vector(df_string)
    print('\n------ End: Calculate tfidf vector -----')

    print('\n----- Start: user projection network ------')
    df_network = calculate_cosine_similarity(df_network, df_string)

    
    df_network.to_pickle(f'{network_save_path}')
    print('\n----- End: user projection network -----')



    print('\n ----- Start: Create twitter client name share graph -----')

    G = create_graph(df_network, 
                 graph_path,
                 campaign_name=new_campaign,
                 source_column='source',
                 target_column='target',
                 weight_column='cosine',
                 type_text=f'tweet_client_{campaign_type}')
    
    print('\n ----- End: Create twitter client name share graph -----')
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_clust = nx.average_clustering(G, weight='cosine')
            
    return [new_campaign, campaign_type, num_nodes, num_edges, avg_clust, len_uniques]

    

