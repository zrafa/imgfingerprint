
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

    // Inicializar ORB
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Detectar y computar descriptores
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;


    float i;
    int j,k,l, m, n;
    int cant, ii, total;
    for (i=1.01; i<=2.0; i=i+0.1) 	// factor de escala
    for (j=1; j<=20; j=j+2) {	// nlevels
		std::cout << "nlevels :" << j << " " << std::endl;
    for (k=5; k<=100; k=k+5)	// edgeThreshold
    for (l=1; l<=2; l++)	// firstLevel
    for (m=2; m<=4; m++)	// WTA_K
    for (n=5; n<=100; n=n+5) {	// patchSize

	orb->setMaxFeatures(50);
	orb->setScaleFactor(i);
	orb->setNLevels(j);
	orb->setEdgeThreshold(k);
	orb->setFirstLevel(l);
	orb->setWTA_K(m);
	orb->setPatchSize(n);

    	orb->detectAndCompute(image, mask, keypoints, descriptors);


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
	if (cant >= (50*total/100)) { 	// if cant es un 70%
		if (en_tronco(orb, image2, mask, keypoints, descriptors, 124, 184, 50)
				&&
		en_tronco(orb, image3, mask, keypoints, descriptors, 144, 184, 50)) 		
		{

		std::cout << "cant:" << cant << " scalefactor " << i 
			<< " nlevels " << j 
			<< " edgethreshold " << k 
			<< " firstlevel " << l 
			<< " wta_k " << m 
			<< " patchsize " << n 
			<< std::endl;
		// Dibujar los puntos clave
		cv::Mat outputImage;
		cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0));

		// Mostrar la imagen con los puntos clave
		cv::imshow("Keypoints", outputImage);
		cv::waitKey(0);
		}
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

