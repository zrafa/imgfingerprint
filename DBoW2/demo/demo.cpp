/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <opencv2/ximgproc.hpp>



#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 229;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}


// ----------------------------------------------------------------------------

void convertir (const char * origen)
{
        char comando[256];
//        sprintf(comando, "convert %s -gravity Center -crop 60%cx100%c+0+0 +repage /tmp/output2.png", origen, '%', '%');

 //      system(comando);

        sprintf(comando, "convert %s   -gravity North -region 100%cx50%c -fill black -colorize 100%c /tmp/output.png", origen, '%', '%', '%');
       system(comando);
}


// ----------------------------------------------------------------------------

int main()
{
  vector<vector<cv::Mat > > features;
  loadFeatures(features);

  testVocCreation(features);

  wait();

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(vector<vector<cv::Mat > > &features)
{
  features.clear();
  features.reserve(NIMAGES);

  //cv::Ptr<cv::ORB> orb = cv::ORB::create(8000);
  cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.11, 17, 95, 1, 2, cv::ORB::HARRIS_SCORE, 70);



  cout << "Extracting ORB features..." << endl;
  for(int i = 0; i < NIMAGES; ++i)
  {
    stringstream ss;
    ss << "i" << i << ".jpg";

    //cv::Mat image = cv::imread(ss.str(), 0);
 //   cv::Mat image = cv::imread(ss.str(), cv::IMREAD_COLOR);

      std::string str = ss.str();
    // Obtener un puntero const char* directamente
	convertir(str.c_str());
//	cv::Mat image = cv::imread("/tmp/output.png", 0);
	cv::Mat image = cv::imread("/tmp/output.png", cv::IMREAD_COLOR);









    cv::Mat mask;
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);



        // Dibujar los puntos clave en la imagen
    cv::Mat image_with_keypoints;
    drawKeypoints(image, keypoints, image_with_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    
    




    // Mostrar la imagen con los puntos clave detectados
   cv::namedWindow("ORB Keypoints", cv::WINDOW_NORMAL);
   cv::imshow("ORB Keypoints", image_with_keypoints);

    // Esperar a que se presione una tecla para cerrar la ventana
  cv::waitKey(0);




    features.push_back(vector<cv::Mat >());
    changeStructure(descriptors, features.back());
  }
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  //const int k = 9;
  //const int L = 3;
  // const int k = 15;
  const int k = 30;
  const int L = 5;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  OrbVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  /*
  BowVector v1, v2;
  // RAFA
  double puntaje[30];
  int i;
  for (i=0; i<30; i++)
	puntaje[i] = 0.0; 
  int h=0;
  int kk=0;

  for(i = 85; i < NIMAGES; i++)
  {
  h=0;
  for (kk=0; kk<30; kk++)
	puntaje[kk] = 0.0; 

    voc.transform(features[i], v1);
    for(int j = 1; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;

      if ((i>=85) && (j<85))
		puntaje[h] += score;
      if ((i>=85) && ((j%5)==0)) {
	      h++;
      }
  }
  if (i>=85) {
  cout << "imagen " << i << "  vs conjuntos "; 
  h=0;
  double cmp=0.0;
  for (kk=0; kk<17; kk++) {
	  if (puntaje[kk] > cmp) {
		  cmp = puntaje[kk];
		  h = kk;
	  }
  }
  cout << " arbol: " << h << endl;
  } /* del if */
  
   // } /* fin del for j */


  cout << "presione una tecla para crear la BD y consultar" << endl;
  getchar();



  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    // db.query(features[i], ret, 4);
    db.query(features[i], ret, 8);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;
  exit(0);




  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;


  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // load the vocabulary from disk
  OrbVocabulary voc("small_voc.yml.gz");
  
  OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  OrbDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


