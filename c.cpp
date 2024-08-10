
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

int main() {
    // Cargar la imagen
    cv::Mat image = cv::imread("tronco4.png", cv::IMREAD_COLOR);
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
		if ((x>=143) && (x<=183))
			cant++;
    	}
	std::cout << " total " << total << " cant " << cant << std::endl;
	if (cant >= (55*total/100)) { 	// if cant es un 70%
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
    }   // del for i

    // Dibujar los puntos clave
    cv::Mat outputImage;
    cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0));

    // Mostrar la imagen con los puntos clave
    cv::imshow("Keypoints", outputImage);
    cv::waitKey(0);

    return 0;
}

