# from make_embeddings_online import make_embeddings
from make_embeddings import make_embeddings, epochs_make_embeddings
from classifier_mlp_AM import perform_prediction

def pipeline4(graph_c, entities_c, list_of_value_headers, test_set_fn, create_emb=True, show_all=True):
    graph_c_name = graph_c.split('/')[-1]
    # entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
    entities_emb = entities_c[:-4] + "_w_emb.csv" #save in original folder, instead of directly in /data
    # entities_emb = "data/entities_dmg777k_TRAIN_w_emb.csv"

    if create_emb:
        make_embeddings(entities_fn=entities_c, 
                        kg_fn=graph_c, 
                        new_entities_fn=entities_emb,
                        # entities_column_name="datetime",reverse=True,
                        entities_column_name="entity",reverse=True,
                        # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
                        show_all=True) #need to show_all, only way to see time it takes to train representations
                        # show_all=show_all)

    final_results = perform_prediction(dataset_fn=entities_emb, show_all=show_all, list_of_value_headers=list_of_value_headers, save_model=True, test_set_fn=test_set_fn)
    # print(final_results["train_result"], final_results["test_result"])
    return final_results

# def epochs_pipeline(epochs, graph_c, entities_c, create_emb=True, show_all=True):
#     graph_c_name = graph_c.split('/')[-1]
#     entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
#     print(entities_emb)
#     # entities_emb = "data/1000_w_random_emb.csv"

#     if create_emb:
#         epochs_make_embeddings(epochs, entities_fn=entities_c, 
#                         kg_fn=graph_c, 
#                         new_entities_fn=entities_emb,
#                         entities_column_name="timestamp",reverse=True,
#                         # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
#                         show_all=True) #need to show_all, only way to see time it takes to train representations
#                         # show_all=show_all)

#     final_results = epochs_perform_predictions(epochs, dataset_fn=entities_emb, show_all=show_all)
#     # print(final_results["train_result"], final_results["test_result"])
#     return final_results


def only_create_embeddings(graph, entities_file, new_entities_file, show_all=True):
    make_embeddings(entities_fn=entities_file, 
                    kg_fn=graph, 
                    new_entities_fn=new_entities_file,
                    # entities_column_name="datetime",reverse=True,
                    entities_column_name="entity",reverse=True,
                    # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
                    show_all=True) #need to show_all, only way to see time it takes to train representations
                    # show_all=show_all)



if __name__ == '__main__':
    am = True
    dmg= False
    md = False
    embeddings_not_features=True
    use_one_dataset=False

    pre_subgraphs = True
    create_embeddings=False

    # entities_file = "data/entity_files/NY_entities_uriTimeStamps.csv"
    if am:
        # new_entities_file = "data/AM_entities_units_COMBINED_w_emb.csv" #for the updating experiment
        entities_file = "data/AM_entities_units.csv" #for the updating experiment
        # entities_file = "data/AM_entities_units_COMBINED.csv"
        list_of_graphs = ["data/amplus-stripped_without_edm_object-.nt"]
        list_of_value_headers=["gram","gram2","gram3","gram4","mm","mm2","mm3","mm4","mm5","mm6","cm","cm2","cm3","cm4","cm5","cm6","cm7","cm8","cm9","cm10","cm11","cm12","cm13","cm14","cm15","cm16","cm17","cm18","cm19","cm20","cm21","cm22","cm23","cm24","cm25","cm26","cm27","cm28","cm29","cm30","ml","gr","gr2","gr3","gr4","gr5","gr6","gr7","gr8","kg","kg2","kg3","G","G2","gr.","m","liter"]
        test_set_fn="data/AM_entities_units_VALIDATION_w_emb.csv"
    elif dmg:    
        entities_file = "data/entities_dmg777k_TRAIN.csv"
        # entities_file = "data/entities_dmg777k_COMBINED.csv"
        list_of_graphs = ["data/dmg777k_stripped_without_thumbs_geo-.nt"]
        list_of_value_headers=["year","population","codeNationalMonument"]
        test_set_fn="data/entities_dmg777k_VALIDATION_w_emb.csv"
    elif md:    
        entities_file = "data/entities_md_raw_TRAIN.csv"
        entities_file = "data/entities_md_raw_COMBINED.csv"
        list_of_graphs = ["data/md_raw_without_Images-.nt"]
        list_of_value_headers=["duration","boxoffice","cost"]
        test_set_fn="data/entities_md_raw_VALIDATION_w_emb.csv"

    if embeddings_not_features:
        list_of_value_headers = []
    if use_one_dataset:
        test_set_fn = ""

    epochs = 20
    walkLength = 2
    nWalks = 25
    reverse = True

    # for part_id in range(100, 101):
    #     multiple_entities_files_pipeline(part_id, epochs, walkLength, nWalks)

    # only_create_embeddings(list_of_graphs[0], "data/entities_md_raw_COMBINED.csv", "data/entities_md_raw_COMBINED_w_emb.csv")

    # final_results = perform_prediction(dataset_fn=entities_emb, show_all=True, save_model=True)


#### PRE SUBGRAPHS
    if pre_subgraphs:
        counter = 0
        for graph in list_of_graphs:
            print(graph, '-', entities_file, "epochs=", epochs, "walkLength=", walkLength, "- nWalks=", nWalks)
            for i in range(0,1):
                # results = walks_pipeline(epochs, graph, entities_file, create_emb=True, show_all=False, walkLength=walkLength, nWalks=nWalks, reverse=reverse)
                results = pipeline4(graph, entities_file, list_of_value_headers, test_set_fn, create_emb=create_embeddings, show_all=True)
                print(str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ','))
                # for k in range(0,2):
                #     # results = walks_pipeline(epochs, graph, entities_file, create_emb=False, show_all=False, walkLength=walkLength, nWalks=nWalks, reverse=reverse)
                #     results = pipeline4(graph, entities_file, create_emb=False, show_all=False)
                #     print(str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ','))

                print("--- finished: ", counter, "---")
                counter = counter+1


########PRE_EPOCHS
    # counter = 1
    # for graph in list_of_graphs:
    #     for i in range(0,1):
    #         # print(graph, '-', entities_file   )
    #         results = pipeline4(graph, entities_file, create_emb=True, show_all=False)
    #         print(results["train_result"], results["test_result"])
    #         # print("targets x:", calc_distribution(results["targets_x"]))
    #         # print("targets y:", calc_distribution(results["targets_y"]))
    #         res = str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',')
    #         for k in range(0,2):
    #             results = pipeline4(graph, entities_file, create_emb=False, show_all=False)
    #             print(results["train_result"], results["test_result"])
    #             # print("targets x:", calc_distribution(results["targets_x"]))
    #             # print("targets y:", calc_distribution(results["targets_y"]))
    #             res += "\t" + str(results["train_result"]).replace('.', ',') +"\t"+ str(results["test_result"]).replace('.', ',')        
    #         print(res)
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         print("--- finished: ", counter, "---")
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         # print("--- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    #         counter = counter+1
    # #         with open("all_results_alldev_timestamp.csv", 'a') as results_file:
    # #             results_file.write(graph+';'+str(results["train_result"])+';'+str(results["test_result"])+'\n')