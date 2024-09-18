#include "utils.hpp"

#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */

void DMU::loadCSV(std::string path, std::map<long int, Eigen::MatrixXd> &content, int cols) {

    // Clear any old data
    content.clear();

    // Open the file
    std::ifstream file;
    std::string line;
    file.open(path);

    // Check that it was successfull
    if (!file) {
        std::cerr << "Unable to open groundtruth file : " << std::endl <<  path.c_str() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Skip the first line as it is just the header
    std::getline(file, line);

    // Loop through each line in the file
    
    while (std::getline(file, line)) {
        int i = 0;
        // Loop variables
        std::istringstream s(line);
        std::string field;
        Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(1, cols);
        // Loop through this line
        while (getline(s, field, ',')) {

        // Save our groundtruth state value
        temp(0, i) = std::atof(field.c_str());
        i++;
        }
        if (i != cols) {
            std::cout << "Invalid line, number of arguments is not correct : " << std::endl << line.c_str() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // Append to our groundtruth map
        content.insert({temp(0, 0), temp});
    }
    

    file.close();
}

DMU::itemList::itemList(std::string path_to_data){
    main_path = path_to_data;
    
    // Construct a string
    std::string item_name;

    int zero_padding_counter = 0;
    for (const auto & item_dir : std::filesystem::directory_iterator(path_to_data)){
        std::string item_path = item_dir.path();
        
        // This while loop assigns the number after the last "\" to splitwise_dir
        std::stringstream ss(item_path); 
        while (std::getline(ss, item_name, '/'));

        // get rid of '.png'
        std::stringstream ss2(item_name); 
        std::getline(ss2, item_name, '.'); 

        // convert the string to integer
        long int item_id = std::stol(item_name);

        // save the image index
        item_ids.push_back(item_id);

        // Check if the name starts with "0"
        if(int(item_name.front()) == 48){ // when you convert char '0' to int, it results in 48
            ++zero_padding_counter;
        }
    }


    // If at least two image name starts with "0"
    // then we have zero-padding in the naming of the images. 
    if(zero_padding_counter>1){
        has_zero_padding = true;
    }else{
        has_zero_padding = false;
    }


    // get the extension of the image
    auto first_item = *std::filesystem::directory_iterator(path_to_data);
    std::filesystem::path path_obj(first_item);
    file_extension = path_obj.extension().string();

    // If we have a zero padding in the naming
    if(has_zero_padding) {
        NUM_ZEROS = item_name.length();
    }
    
    // sort the indexes so that they are in proper order. (Smallest index comes first)
    std::sort(item_ids.begin(), item_ids.end());
}


std::string DMU::itemList::getItemPath(int img_idx){
    std::string item_id_string = std::to_string(item_ids.at(img_idx));
    std::string path_to_item;
    if(has_zero_padding){
        path_to_item = std::filesystem::path(main_path) / (std::string(NUM_ZEROS - item_id_string.length(), '0') + item_id_string + file_extension);
    }
    else{
        path_to_item = std::filesystem::path(main_path) / (item_id_string + file_extension);
    }
    return path_to_item;
}


std::string DMU::itemList::getItemName(int img_idx){
    std::string item_id_string = std::to_string(item_ids.at(img_idx));
    std::string item_name;
    if(has_zero_padding){
        item_name = (std::string(NUM_ZEROS - item_id_string.length(), '0') + item_id_string);
    }
    else{
        item_name = (item_id_string);
    }
    return item_name;
}

std::string DMU::itemList::getItemPathFromName(std::string item_name){
    std::string path_to_item;
    path_to_item = std::filesystem::path(main_path) / (item_name + file_extension);
    return path_to_item;
}

DMU::dataHandler::dataHandler(std::string main_path):
    flow_folder((std::filesystem::path(main_path) / flow_folder_name).string()),
    depth_folder((std::filesystem::path(main_path) / depth_folder_name).string()),
    image_folder((std::filesystem::path(main_path) / image_folder_name).string()),
    path_to_gt((std::filesystem::path(main_path) / gt_file_name).string()),
    path_to_imu((std::filesystem::path(main_path) / imu_file_name).string()),
    flowList(flow_folder),
    depthList(depth_folder),
    imgList(image_folder)
    {   
        // Load GT
        loadCSV(path_to_gt, gt_data, 16);

        // Load Preintegrated IMU measurement
        loadCSV(path_to_imu, imu_data, 16);
    }


cv::Mat DMU::load_flow(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return cv::Mat();
    }

    // Read the magic number
    float magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(float));

    if (magic != 202021.25) {
        std::cerr << "Invalid magic number: " << magic << std::endl;
        return cv::Mat();
    }

    // Read width and height
    int32_t width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&height), sizeof(int32_t));

    // Read flow data
    std::vector<float> data(height * width * 2);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    // Reshape to a 3D cv::Mat
    cv::Mat flow(height, width, CV_32FC2, data.data());

    return flow.clone(); // Ensure the data is copied to the cv::Mat
}

// Generate the color wheel based on the algorithm by Baker et al.
void VU::makeColorWheel(std::vector<cv::Vec3b> &colorwheel) {
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;

    int i;

    for (i = 0; i < RY; i++) colorwheel.push_back(cv::Vec3b(255, 255 * i / RY, 0));
    for (i = 0; i < YG; i++) colorwheel.push_back(cv::Vec3b(255 - 255 * i / YG, 255, 0));
    for (i = 0; i < GC; i++) colorwheel.push_back(cv::Vec3b(0, 255, 255 * i / GC));
    for (i = 0; i < CB; i++) colorwheel.push_back(cv::Vec3b(0, 255 - 255 * i / CB, 255));
    for (i = 0; i < BM; i++) colorwheel.push_back(cv::Vec3b(255 * i / BM, 0, 255));
    for (i = 0; i < MR; i++) colorwheel.push_back(cv::Vec3b(255, 0, 255 - 255 * i / MR));
}

// Convert optical flow to color based on the color wheel
cv::Mat VU::flowToColor(const cv::Mat &flow_in) {
    cv::Mat flow;

    // Flow is assumed to be float
    flow_in.convertTo(flow, CV_32F);

    cv::Mat color(flow.size(), CV_8UC3);
    std::vector<cv::Vec3b> colorwheel;
    makeColorWheel(colorwheel);
    int ncols = colorwheel.size();

    for (int y = 0; y < flow.rows; y++) {
        for (int x = 0; x < flow.cols; x++) {
            
            cv::Vec2f flow_at_xy = flow.at<cv::Vec2f>(y, x);
            float fx = flow_at_xy[0];
            float fy = flow_at_xy[1];
            float rad = sqrt(fx * fx + fy * fy);
            float angle = atan2(-fy, -fx) / CV_PI;
            float fk = (angle + 1.0) / 2.0 * (ncols - 1);
            int k0 = (int)fk;
            int k1 = (k0 + 1) % ncols;
            float f = fk - k0;

            cv::Vec3b col0 = colorwheel[k0];
            cv::Vec3b col1 = colorwheel[k1];
            cv::Vec3b col;

            for (int b = 0; b < 3; b++)
                col[b] = (uchar)((1 - f) * col0[b] + f * col1[b]);

            if (rad <= 1.0)
                col *= rad;
            else
                col *= 1.0;
            color.at<cv::Vec3b>(y, x) = col;
            
        }
    }
    return color;
}


cv::Mat VU::warpFlow(const cv::Mat& img, const cv::Mat& flow) {
    // Get the height and width from the flow
    int h = flow.rows;
    int w = flow.cols;

    // Create a new flow matrix
    cv::Mat flowRemap = flow.clone();

    // Datatype should be CV_32F
    flowRemap.convertTo(flowRemap, CV_32F);

    // Negate the flow
    flowRemap *= -1;

    // Adjust flow for remapping
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            flowRemap.at<cv::Vec2f>(y, x)[0] += (float)x; // Add x coordinate
            flowRemap.at<cv::Vec2f>(y, x)[1] += (float)y; // Add y coordinate
        }
    }

    // Remap the image using the flow
    cv::Mat warpedImg;
    cv::remap(img, warpedImg, flowRemap, cv::Mat(), cv::INTER_LINEAR);

    return warpedImg;
}



std::vector<cv::Mat> VU::depthErrorHistogram(cv::Mat depth_map, cv::Mat estimated_depth_map, long int totalPixCount){

    // Copmute depth estimation error
    cv::Mat depth_map_err = estimated_depth_map - depth_map;

    // Compute percentage error
    cv::Mat depth_map_err_percentage;
    cv::divide(depth_map_err, depth_map, depth_map_err_percentage);
    depth_map_err_percentage = depth_map_err_percentage * 100;

    // Define the number of bins
    int histSize = 200; // Number of intensity levels for grayscale image

    // Set the ranges for pixel values (0-255 for grayscale)
    float range[] = { -10, 10 };
    const float* histRange = { range };

    // Create a cv::Mat to store the histogram
    cv::Mat hist_err, hist_err_perc;

    // Calculate the histogram
    cv::calcHist(&depth_map_err, 1, 0, cv::Mat(), hist_err, 1, &histSize, &histRange);
    cv::calcHist(&depth_map_err_percentage, 1, 0, cv::Mat(), hist_err_perc, 1, &histSize, &histRange);

    // Convert the histogram data to a std::vector for matplotlibcpp
    std::vector<float> histData_err(histSize);
    for (int i = 0; i < histSize; i++) {
        histData_err[i] = hist_err.at<float>(i) / ((float) totalPixCount);
    }

    // Convert the histogram data to a std::vector for matplotlibcpp
    std::vector<float> histData_err_perc(histSize);
    for (int i = 0; i < histSize; i++) {
        histData_err_perc[i] = hist_err_perc.at<float>(i) / ((float) totalPixCount);
    }

    // Prepare x-axis values for the histogram
    std::vector<float> xValues(histSize);
    for (int i = 0; i < histSize; i++) {
        xValues[i] = (i * (range[1] - range[0]) / histSize) + range[0];
    }

    // Plot the histogram using matplotlibcpp
    plt::close();
    plt::figure_size(640, 512); // Set figure size
    plt::plot(xValues,histData_err);
    plt::title("Depth Error Histogram");
    plt::xlabel("Error");
    plt::ylabel("PDF");
    plt::ylim(0.0, 0.5);
    plt::grid(true); // Add grid for better visibility
    
    // Render the plot to a buffer
    plt::save("DepthErrorHist.png"); // Save to a temporary file
    cv::Mat depth_err_hist_img = cv::imread("DepthErrorHist.png"); // Load the saved image

    // cv::imshow("Depth Error Histogram", histogramImage);

    plt::clf();
    plt::close();

    plt::figure_size(640, 512); // Set figure size
    plt::plot(xValues,histData_err_perc);
    plt::title("Depth Error Percentage Histogram");
    plt::xlabel("Error / Actual * 100");
    plt::ylabel("PDF");
    plt::ylim(0.0, 0.2);
    plt::grid(true); // Add grid for better visibility
    
    // Render the plot to a buffer
    plt::save("DepthErrorPercentageHist.png"); // Save to a temporary file
    cv::Mat depth_err_perc_hist_img = cv::imread("DepthErrorPercentageHist.png"); // Load the saved image

    std::vector<cv::Mat> histograms;
    histograms.push_back(depth_err_hist_img);
    histograms.push_back(depth_err_perc_hist_img);

    // Close opened windows
    plt::close(); 

    return histograms;
}

Eigen::Matrix3d LU::Skew(Eigen::Vector3d vec){
    Eigen::Matrix3d S;
    S <<    0.0, -vec[2], vec[1],
            vec[2], 0.0, -vec[0],
            -vec[1], vec[0], 0.0;
    return S;
}

Eigen::Matrix3d LU::exp_SO3(Eigen::Vector3d psi){
    double angle = psi.norm();

    // If psi is too small, return Identity
    if(angle<LU::_tolerance){
        return Eigen::MatrixXd::Identity(3,3);
    }

    Eigen::Vector3d unit_psi        = psi / angle;
    Eigen::Matrix3d unit_psi_skew   = LU::Skew(unit_psi);

    Eigen::Matrix3d R = Eigen::MatrixXd::Identity(3,3) + std::sin(angle) * unit_psi_skew + (1-std::cos(angle)) * unit_psi_skew * unit_psi_skew;

    return R;
}