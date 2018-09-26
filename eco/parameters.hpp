// Set the value the same as testing_ECO_gpu.m
#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#ifdef USE_CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#endif

#include <vector>
#include <string>
#include <opencv2/core.hpp>

#define INF 0x7f800000 //0x7fffffff

using std::string;
using std::vector;

namespace eco
{
// ECO feature[Num_features][Dimension_of_the_feature];
typedef std::vector<std::vector<cv::Mat>> ECO_FEATS;
typedef cv::Vec<float, 2> COMPLEX; // represent a complex number;

// cnn   feature   configuration =========================================
#ifdef USE_CAFFE
struct CnnParameters
{
	string proto = "model/imagenet-vgg-m-2048.prototxt";
	string model = "model/VGG_CNN_M_2048.caffemodel";
	string mean_file = "model/VGG_mean.binaryproto";

	boost::shared_ptr<caffe::Net<float>> net;
	cv::Mat deep_mean_mat, deep_mean_mean_mat;

	string nn_name = "imagenet-vgg-m-2048.mat";
	vector<int> stride = {2, 16};			// stride in total
	vector<int> cell_size = {4, 16};		// downsample_factor
	vector<int> output_layer = {3, 14};		// Which layers to use
	vector<int> downsample_factor = {2, 1}; // How much to downsample each output layer
	int input_size_scale = 1;				// Extra scale factor of the input samples to the network (1 is no scaling)
	vector<int> nDim = {96, 512};			// Original dimension of features (ECO Paper Table 1)
	vector<int> compressed_dim = {16, 64};  // Compressed dimensionality of each output layer (ECO Paper Table 1)
	vector<float> penalty = {0, 0};

	vector<int> start_ind = {3, 3, 1, 1};	 // sample feature start index
	vector<int> end_ind = {106, 106, 13, 13}; // sample feature end index
};
struct CnnFeatures
{
	CnnParameters fparams;
	cv::Size img_input_sz = cv::Size(224, 224); // VGG default input sample size
	cv::Size img_sample_sz;						// the size of sample
	cv::Size data_sz_block0, data_sz_block1;
	cv::Mat mean;
};
#endif

// hog parameters cofiguration =========================================
struct HogParameters
{
	int cell_size = 6;
	int compressed_dim = 10; // Compressed dimensionality of each output layer (ECO Paper Table 1)
	int nOrients = 9;
	size_t nDim = 31; // Original dimension of feature
	float penalty = 0;
};
struct HogFeatures
{
	HogParameters fparams;
	cv::Size img_input_sz;  // input sample size
	cv::Size img_sample_sz; // the size of sample
	cv::Size data_sz_block0;
};

// CN parameters configuration =========================================
struct ColorspaceParameters
{
	string colorspace = "gray";
	int cell_size = 1;
};
struct ColorspaceFeatures
{
	ColorspaceParameters fparams;
	cv::Size img_input_sz;  
	cv::Size img_sample_sz;
	cv::Size data_sz_block0;
};
//---------------------------
struct CnParameters // only used for Color image
{
	string tablename = "look_tables/CNnorm.txt";
	float table[32768][10];
	int cell_size = 4;
	int compressed_dim = 3;
	size_t nDim = 10; 
	float penalty = 0;
};
struct CnFeatures
{
	CnParameters fparams;
	cv::Size img_input_sz; 
	cv::Size img_sample_sz; 
	cv::Size data_sz_block0;
};
//---------------------------
struct IcParameters // only used for gray image
{
	string tablename = "look_tables/intensityChannelNorm6";
	float table[256][5];
	int cell_size = 4;
	int compressed_dim = 3;
	size_t nDim = 5; 
	float penalty = 0;
};
struct IcFeatures
{
	IcParameters fparams;
	cv::Size img_input_sz;
	cv::Size img_sample_sz;
	cv::Size data_sz_block0;
};

// Cojugate Gradient Options Structure =====================================
struct CgOpts 
{
	bool debug;
	bool CG_use_FR;
	float tol;
	bool CG_standard_alpha;
	float init_forget_factor;
	int maxit;
};

// Parameters set exactly the same as 'testing_ECO_HC.m'====================
struct EcoParameters
{
	// Features
	bool useDeepFeature = false;
	bool useHogFeature = true;
	bool useColorspaceFeature = false;// not implemented yet
	bool useCnFeature = false;
	bool useIcFeature = true;

#ifdef USE_CAFFE
	CnnFeatures cnn_features;
#endif
	HogFeatures hog_features;
	ColorspaceFeatures colorspace_feature;
	CnFeatures cn_features;
	IcFeatures ic_features;
	
	// extra parameters
	CgOpts CG_opts;
	float max_score_threshhold = 0.1;

	// Global feature parameters1s
	int normalize_power = 2;
	bool normalize_size = true;
	bool normalize_dim = true;

	// img sample parameters
	string search_area_shape = "square"; // The shape of the samples
	float search_area_scale = 4.0;		 // The scaling of the target size to get the search area
	int min_image_sample_size = 22500;   // Minimum area of image samples, 200x200
	int max_image_sample_size = 40000;   // Maximum area of image samples, 250x250

	// Detection parameters
	int refinement_iterations = 1; // Number of iterations used to refine the resulting position in a frame
	int newton_iterations = 5;	 // The number of Newton iterations used for optimizing the detection score
	bool clamp_position = false;   // Clamp the target position to be inside the image

	// Learning parameters
	float output_sigma_factor = 1.0f / 16.0f; // Label function sigma
	float learning_rate = 0.009; // Learning rate
	size_t nSamples = 30; // Maximum number of stored training samples
	string sample_replace_strategy = "lowest_prior"; // Which sample to replace when the memory is full
	bool lt_size = 0; // The size of the long - term memory(where all samples have equal weight)
	int train_gap = 5; // The number of intermediate frames with no training(0 corresponds to training every frame)
	int skip_after_frame = 10; // After which frame number the sparse update scheme should start(1 is directly)
	bool use_detection_sample = true; // Use the sample that was extracted at the detection stage also for learning

	// Factorized convolution parameters
	bool use_projection_matrix = true;	// Use projection matrix, i.e. use the factorized convolution formulation
	bool update_projection_matrix = true; // Whether the projection matrix should be optimized or not
	string proj_init_method = "pca"; // Method for initializing the projection matrix
	float projection_reg = 1e-7; // Regularization paremeter of the projection matrix (lambda)

	// Generative sample space model parameters
	bool use_sample_merge = true; // Use the generative sample space model to merge samples
	string sample_merge_type = "Merge"; // Strategy for updating the samples
	string distance_matrix_update_type = "exact"; // Strategy for updating the distance matrix

	// Conjugate Gradient parameters
	int CG_iter = 5; // The number of Conjugate Gradient iterations in each update after the first frame
	int init_CG_iter = 10 * 15; // The total number of Conjugate Gradient iterations used in the first frame
	int init_GN_iter = 10; // The number of Gauss-Newton iterations used in the first frame(only if the projection matrix is updated)
	bool CG_use_FR = false; // Use the Fletcher-Reeves(true) or Polak-Ribiere(false) formula in the Conjugate Gradient
	bool CG_standard_alpha = true;  // Use the standard formula for computing the step length in Conjugate Gradient
	int CG_forgetting_rate = 50; // Forgetting rate of the last conjugate direction
	float precond_data_param = 0.75; // Weight of the data term in the preconditioner
	float precond_reg_param = 0.25; // Weight of the regularization term in the preconditioner
	int precond_proj_param = 40; // Weight of the projection matrix part in the preconditioner

	// Regularization window parameters
	bool use_reg_window = true; // Use spatial regularization or not
	double reg_window_min = 1e-4; // The minimum value of the regularization window
	double reg_window_edge = 10e-3; // The impact of the spatial regularization
	size_t reg_window_power = 2; // The degree of the polynomial to use(e.g. 2 is a quadratic window)
	float reg_sparsity_threshold = 0.05; // A relative threshold of which DFT coefficients that should be set to zero

	// Interpolation parameters
	string interpolation_method = "bicubic"; // The kind of interpolation kernel
	float interpolation_bicubic_a = -0.75;   // The parameter for the bicubic interpolation kernel
	bool interpolation_centering = true; // Center the kernel at the feature sample
	bool interpolation_windowing = false; // Do additional windowing on the Fourier coefficients of the kernel

	// Scale parameters for the translation model
	// Only used if: use_scale_filter = false
	size_t number_of_scales = 7; // Number of scales to run the detector
	float scale_step = 1.01f; // The scale factor
	float min_scale_factor;
	float max_scale_factor;

	// Scale filter parameters
	// Only used if: use_scale_filter = true
	bool use_scale_filter = false; // Use the fDSST scale filter or not (for speed)
	float scale_sigma_factor = 1.0f / 16.0f; // Scale label function sigma
	float scale_learning_rate = 0.025;		 // Scale filter learning rate
	int number_of_scales_filter = 17;		 // Number of scales
	int number_of_interp_scales = 33;		 // Number of interpolated scales
	float scale_model_factor = 1.0;			 // Scaling of the scale model
	float scale_step_filter = 1.02;			 // The scale factor for the scale filter
	float scale_model_max_area = 32 * 16;	 // Maximume area for the scale sample patch
	string scale_feature = "HOG4";			 // Features for the scale filter (only HOG4 supported)
	int s_num_compressed_dim = 17;	 // Number of compressed feature dimensions in the scale filter
	float lambda = 1e-2;					 // Scale filter regularization
	float do_poly_interp = true;			 // Do 2nd order polynomial interpolation to obtain more accurate scale
	cv::Size scale_model_sz;


	bool debug = 0; // to show heatmap or not

	// GPU
	bool use_gpu = true; // whether Caffe use gpu or not
	int gpu_id = 0;
};
} // namespace eco
#endif
