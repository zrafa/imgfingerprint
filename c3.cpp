
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


    int cant, total, ii;



    // Ajuste sistemático de los parámetros de BRISK
    for (int thresh = 5; thresh <= 70; thresh += 5) {         // Rango y pasos para el umbral
        for (int octaves = 1; octaves <= 5; octaves++) {         // Rango y pasos para octavas
            for (float patternScale = 0.0f; patternScale <= 2.0f; patternScale += 0.1f) { // Rango y pasos para escala del patrón

                // Crear el detector BRISK con los parámetros específicos
                cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(thresh, octaves, patternScale);

                // Detectar y calcular descriptores
                brisk->detectAndCompute(image, cv::noArray(), keypoints, descriptors);

                // Mostrar resultados
                //std::cout << "BRISK detectó " << keypoints.size() << " keypoints con los siguientes parámetros:\n"
                 //         << "Threshold: " << thresh << ", Octaves: " << octaves
                  //        << ", PatternScale: " << patternScale << std::endl;






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
	if ((total != 0) && (cant >= (50*total/100))) { 	// if cant es un 70%
		//if (en_tronco(orb, image2, mask, keypoints, descriptors, 124, 184, 50)
		//		&&
		//en_tronco(orb, image3, mask, keypoints, descriptors, 144, 184, 50)) 		
		//{

                    // Mostrar o procesar resultados según necesidad
                    std::cout << "brisk: " << thresh 
                              << ", octaves " << octaves
                              << ", pattern scale " << patternScale
                              << std::endl;


		// Dibujar los puntos clave
		cv::Mat outputImage;
		cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0));

		// Mostrar la imagen con los puntos clave
		cv::imshow("Keypoints", outputImage);
		cv::waitKey(0);
		//}
	}


            }
        }
    }

    // Dibujar los puntos clave
    cv::Mat outputImage;
    cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar(0, 255, 0));

    // Mostrar la imagen con los puntos clave
    cv::imshow("Keypoints", outputImage);
    cv::waitKey(0);

    return 0;
}

