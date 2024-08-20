
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

int en_tronco(cv::Ptr<cv::ORB> orb, const cv::Mat& image, const cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, int x1, int x2, int porc) {
	int cant, total;
	int ii;

    // Usar el ORB para detectar y computar los descriptores
    orb->detectAndCompute(image, mask, keypoints, descriptors);

	cant=0;
	total=0;
	// Recorrer los keypoints y obtener sus coordenadas
    	for (ii = 0; ii < keypoints.size(); ++ii) {
		total++;
        	float x = keypoints[ii].pt.x;
		if ((x>=x1) && (x<=x2))
			cant++;
    	}
	std::cout << x1 << " total " << total << " cant " << cant << std::endl;
	if (cant >= (porc*total/100)) 
		return 1;

	return 0;
}


int main() {
    // Cargar la imagen
    cv::Mat image = cv::imread("tronco5.png", cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread("tronco2.jpg", cv::IMREAD_COLOR);
    cv::Mat image3 = cv::imread("tronco4.png", cv::IMREAD_COLOR);

    if (image.empty()) {
        std::cerr << "No se pudo cargar la imagen." << std::endl;
        return -1;
    }

    cv::Mat mask;

    // Inicializar SIFT
//    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // Detectar y computar descriptores
 //   std::vector<cv::KeyPoint> keypoints;
  //  cv::Mat descriptors;

    	std::vector<cv::KeyPoint> keypoints;
    	cv::Mat descriptors;

    double j, k, l;
    int i, m, n;
    int cant, ii, total;
    for (i=1; i<=6; i++) 	// nOctaveLayers
    for (j=0.01; j<=0.1; j=j+0.01) {	// contrastThreshold 
		std::cout << "contrastThreshold:" << j << " " << std::endl;
    for (k=5.0; k<=20.0; k=k+0.5)	// edgeThreshold
    for (l=1.2; l<=2.0; l=l+0.1) {	// sigma

        cv::Ptr<cv::SIFT> sift = cv::SIFT::create(200, i, j, k, l);

    	sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

	cant=0;
	total=0;
	// Recorrer los keypoints y obtener sus coordenadas
    	for (ii = 0; ii < keypoints.size(); ++ii) {
		total++;
        	float x = keypoints[ii].pt.x;
		//if ((x>=120) && (x<=185))
		if ((x>=126) && (x<=165))
			cant++;
    	}
	std::cout << " total " << total << " cant " << cant << std::endl;
	if (cant >= (80*total/100)) { 	// if cant es un 70%
	//	if (en_tronco(orb, image2, mask, keypoints, descriptors, 124, 184, 70))
			//	&&
		//en_tronco(orb, image3, mask, keypoints, descriptors, 144, 184, 40)) 		{
	//	{

		std::cout << "cant:" << cant << " octave " << i 
			<< " contrastthreshold " << j 
			<< " edgethreshold " << k 
			<< " sigma " << l 
			<< std::endl;
		// Dibujar los puntos clave
		cv::Mat outputImage;
		cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0));

		// Mostrar la imagen con los puntos clave
		cv::imshow("Keypoints", outputImage);
		cv::waitKey(0);
	//	}
	}
	      

    }
    }   // del for i

    // Dibujar los puntos clave
    cv::Mat outputImage;
    cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0));

    // Mostrar la imagen con los puntos clave
    cv::imshow("Keypoints", outputImage);
    cv::waitKey(0);

    return 0;
}

