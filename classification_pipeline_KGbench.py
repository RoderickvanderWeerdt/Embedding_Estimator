# from make_embeddings_online import make_embeddings
from make_embeddings import make_embeddings, epochs_make_embeddings
from classifier_mlp_KGbench import perform_prediction
import time

def pipeline4(graph_c, entities_c, list_of_value_headers, test_set_fn, create_emb=True, show_all=True):
    graph_c_name = graph_c.split('/')[-1]
    # entities_emb = "data/" + graph_c_name[:-4] + "_w_emb.csv"
    entities_emb = entities_c[:-4] + "_w_emb.csv" #save in original folder, instead of directly in /data
    # entities_emb = "data/entities_dmg777k_TRAIN_w_emb.csv"

    start_time = time.time()

    if create_emb:
        make_embeddings(entities_fn=entities_c, 
                        kg_fn=graph_c, 
                        new_entities_fn=entities_emb,
                        # entities_column_name="datetime",reverse=True,
                        entities_column_name="entity",reverse=True,
                        # entities_column_name="https://interconnectproject.eu/example/DEKNres4_HP_URI",reverse=True,
                        show_all=True) #need to show_all, only way to see time it takes to train representations
                        # show_all=show_all)
    elif list_of_value_headers != []:
        entities_emb = entities_c #use the file without embeddings
    emb_time = time.time() - start_time
    start_time = time.time()
    final_results = perform_prediction(dataset_fn=entities_emb, show_all=show_all, list_of_value_headers=list_of_value_headers, save_model=True, test_set_fn=test_set_fn)
    print(emb_time, '---', time.time() - start_time)
    # print(final_results["train_result"], final_results["test_result"])
    return final_results


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
    am = False
    dmg= True
    md = False
    toy = False
    OfficeGraph = False
    opsd = False

    embeddings_not_features=True
    use_one_dataset=True

    pre_subgraphs = True
    create_embeddings=False
    if not embeddings_not_features: create_embeddings = False

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
    elif toy:
        noise = 10
        entities_file = "data/entities_toy_saref-"+str(noise)+","+str(noise)+".csv"
        list_of_graphs = ["data/toy_saref-"+str(noise)+","+str(noise)+".nt"]
        list_of_value_headers=["values_dev0","values_dev1","values_dev2","values_dev3","values_dev4","values_dev5","values_dev6","values_dev7","values_dev8","values_dev9"]
    elif OfficeGraph:
        list_of_graphs = ["data/OfficeGraph/OfficeGraph_res1devA_enriched.nt"]
        entities_file = "data/OfficeGraph/entities_w_values.csv"
        list_of_value_headers=["https://interconnectproject.eu/example/property_R5_2__humidity_","https://interconnectproject.eu/example/property_R5_2__temp_","https://interconnectproject.eu/example/property_R5_2__co2_"]
        # entities_file = "data/OfficeGraph/entities_w_ALL_values.csv" #trained on all devices on 7th floor
        # list_of_value_headers=["https://interconnectproject.eu/example/property_R5_154__co2_", "https://interconnectproject.eu/example/property_R5_154__humidity_", "https://interconnectproject.eu/example/property_R5_154__temp_", "https://interconnectproject.eu/example/property_R5_180__co2_", "https://interconnectproject.eu/example/property_R5_180__humidity_", "https://interconnectproject.eu/example/property_R5_180__temp_", "https://interconnectproject.eu/example/property_R5_211__co2_", "https://interconnectproject.eu/example/property_R5_211__humidity_", "https://interconnectproject.eu/example/property_R5_211__temp_", "https://interconnectproject.eu/example/property_R5_2__co2_", "https://interconnectproject.eu/example/property_R5_2__humidity_", "https://interconnectproject.eu/example/property_R5_2__temp_", "https://interconnectproject.eu/example/property_R5_95__co2_", "https://interconnectproject.eu/example/property_R5_95__humidity_", "https://interconnectproject.eu/example/property_R5_95__temp_", "https://interconnectproject.eu/example/property_Zigbee_Thermostat_9__battery_lvl_", "https://interconnectproject.eu/example/property_Zigbee_Thermostat_9__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_26__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_22__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_22__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_26__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_26__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_47__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_47__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_63__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_63__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_68__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_68__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_86__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_86__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_87__battery_lvl_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_87__status_", "https://interconnectproject.eu/example/property_Zigbee_Thermostat_9__status_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_26__contact_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_68__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_86__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_63__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_22__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_47__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_87__temp_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_87__contact_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_47__contact_", "https://interconnectproject.eu/example/property_Zigbee_Thermostat_9__heating_setpoint_", "https://interconnectproject.eu/example/property_SmartSense_Multi_Sensor_22__contact_"]
    elif opsd:
        list_of_graphs = ["data/OPSD/opsd_res1devA_enriched.nt"]
        entities_file = "data/OPSD/opsd_entities_bin.csv" #trained on all devices on 7th floor
        # entities_file = "data/OPSD/entities_opsd_res1devA_enriched_TransE_embs_bin.csv" #used TransE
        list_of_value_headers=["https://interconnectproject.eu/example/DEKNres2_GI", "https://interconnectproject.eu/example/DEKNres3_GE", "https://interconnectproject.eu/example/DEKNres4_GE", "https://interconnectproject.eu/example/DEKNres3_GI", "https://interconnectproject.eu/example/DEKNres3_CP", "https://interconnectproject.eu/example/DEKNres3_WM", "https://interconnectproject.eu/example/DEKNres3_FR", "https://interconnectproject.eu/example/DEKNres5_DW", "https://interconnectproject.eu/example/DEKNres5_RF", "https://interconnectproject.eu/example/DEKNres3_RF", "https://interconnectproject.eu/example/DEKNres6_WM", "https://interconnectproject.eu/example/DEKNres2_WM", "https://interconnectproject.eu/example/DEKNres6_DW", "https://interconnectproject.eu/example/DEKNres1_GI", "https://interconnectproject.eu/example/DEKNres6_FR", "https://interconnectproject.eu/example/DEKNres4_PV", "https://interconnectproject.eu/example/DEKNres4_WM", "https://interconnectproject.eu/example/DEKNres1_HP", "https://interconnectproject.eu/example/DEKNres1_WM", "https://interconnectproject.eu/example/DEKNres1_PV", "https://interconnectproject.eu/example/DEKNres3_DW", "https://interconnectproject.eu/example/DEKNres4_DW", "https://interconnectproject.eu/example/DEKNres3_PV", "https://interconnectproject.eu/example/DEKNres1_FR", "https://interconnectproject.eu/example/DEKNres5_GI", "https://interconnectproject.eu/example/DEKNres6_GI", "https://interconnectproject.eu/example/DEKNres2_CP", "https://interconnectproject.eu/example/DEKNres6_PV", "https://interconnectproject.eu/example/DEKNres2_DW", "https://interconnectproject.eu/example/DEKNres6_CP", "https://interconnectproject.eu/example/DEKNres5_WM", "https://interconnectproject.eu/example/DEKNres4_HP", "https://interconnectproject.eu/example/DEKNres1_DW", "https://interconnectproject.eu/example/DEKNres4_FR", "https://interconnectproject.eu/example/DEKNres4_GI", "https://interconnectproject.eu/example/DEKNres4_EV"]

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