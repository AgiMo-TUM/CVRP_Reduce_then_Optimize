# import sys
# from pathlib import Path

# repo_root = Path(__file__).resolve().parents[1]   # adjust depth if needed
# sys.path.append(str(repo_root))


from scripts.analyses_04.visualization import _generate_runtime_missing_arc_plot_impl
from scripts.analyses_04.visualization import _generate_runtime_missing_arc_plot_impl_knn_compare
from scripts.analyses_04.visualization import _generate_runtime_missing_arc_plot_impl_TW
from scripts.analyses_04.visualization import generate_runtime_missing_arc_plot_with_size_bars
from scripts.analyses_04.visualization import generate_new_HGS_77_reduction_plot_2
from scripts.analyses_04.visualization import plt_new_HGS_77_average_runtime_2

from scripts.analyses_04.visualization import get_performances_selection_method

# _generate_runtime_missing_arc_plot_impl(r"benchmarking_XML_test_top_k_gnn_exp_0355\ml-reduction\top_k\exact-grb_timeout_60\Test_benchmarking_XML_top_k_gnn_test_exp_0355")
# _generate_runtime_missing_arc_plot_impl_knn_compare(r"benchmarking_XML_test_top_k_gnn_exp_0355\ml-reduction\top_k\exact-grb_timeout_60\Test_benchmarking_XML_top_k_gnn_test_exp_0355",
#                                                     r"benchmarking_XML_test_distance_knn_exp_0355\ml-reduction\distance_knn\exact-grb_timeout_60\Test_benchmarking_XML_distance_knn_test_exp_0355")
_generate_runtime_missing_arc_plot_impl_knn_compare(r"benchmarking_XML_TW_1_test_top_k_gnn_exp_0157\ml-reduction\top_k\exact-grb_timeout_60\Test_benchmarking_XML_TW_1_top_k_gnn_test_exp_0157",
                                                    r"benchmarking_XML_TW_1_test_distance_knn_exp_0157\ml-reduction\distance_knn\exact-grb_timeout_60\Test_benchmarking_XML_TW_1_distance_knn_test_exp_0157")
# _generate_runtime_missing_arc_plot_impl_TW(r"benchmarking_XML_TW_1_test_size_thres_direct_exp_0157\ml-reduction\size\exact-grb_timeout_60\Test_benchmarking_XML_TW_1_size_thres_test_direct_exp_0157")
# generate_runtime_missing_arc_plot_with_size_bars(r"benchmarking_XML_new_code_test_probability\ml-reduction\prob\exact-grb_timeout_60\Test_benchmarking_XML_exp_0355_test_new_code_probability5000")
# generate_new_HGS_77_reduction_plot_2([r"benchmarking_big_XML_test_top_k_exp_0006_hgs_prune_True_100\ml-reduction\top_k\hgs-heu_time_100\Test_benchmarking_big_XML_top_k_test_exp_0006_hgs_prune_True_100",
# r"benchmarking_big_XML_test_top_k_exp_0006_hgs_prune_True_200\ml-reduction\top_k\hgs-heu_time_100\Test_benchmarking_big_XML_top_k_test_exp_0006_hgs_prune_True_200"],
#                                      "hgs_dict/output.pkl")
# plt_new_HGS_77_average_runtime_2("hgs_dict/output.pkl")

# get_performances_selection_method(r"trained_models_BCE_loss_XML_exp_0355\model_gcnn_features_graph_additional_3_prediction_task_binary_classification_normalization_standard_hidden_layer_dim_64_num_conv_layers_4_num_dense_layers_1\application\best_checkpoint.pth.tar",
                                #    r"data\XML_split\test", False, 0, 500, "distance_knn", [30, 20, 16, 8, 4, 2] )