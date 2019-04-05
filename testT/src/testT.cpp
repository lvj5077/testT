//
// Created by jin on 4/1/19.
//

#include "SLAMbase.h"

#include <algorithm>
#include <iostream>
using namespace std;

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;


#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

int main( int argc, char** argv )
{
    int depthL = 180;
    int depthH = 3000;
    // 相机内参
    CAMERA_INTRINSIC_PARAMETERS C;
    C.cx = 315.40594482421875;
    C.cy = 244.33926391601562;
    C.fx = 614.8074340820312;
    C.fy = 614.5072021484375;
    C.scale = 1000.0;
    double camera_matrix_data[3][3] = {
        {C.fx, 0, C.cx},
        {0, C.fy, C.cy},
        {0, 0, 1}
    };

    // 构建相机矩阵
    cv::Mat cameraMatrix( 3, 3, CV_64F, camera_matrix_data );

    // 第一个帧的三维点
    vector<cv::Point3f> firstPts;
    // 第二个帧的三维点
    vector<cv::Point3f> secondPts;
    // 第二个帧的图像点
    vector< cv::Point2f > pts_img;


    // // checked // still camera in feature rich environment
    // cv::Mat rgb1 = cv::imread( "/home/jin/Data/04_04_2019/sCamSscene/color/1554395077.593656341.png");
    // cv::Mat rgb2 = cv::imread( "/home/jin/Data/04_04_2019/sCamSscene/color/1554395077.869748494.png");
    // cv::Mat depth1 = cv::imread( "/home/jin/Data/04_04_2019/sCamSscene/aligned_depth/1554395077.593656341.png", -1);
    // cv::Mat depth2 = cv::imread( "/home/jin/Data/04_04_2019/sCamSscene/aligned_depth/1554395077.869748494.png", -1);


    // // checked // still camera in normal environment
    // cv::Mat rgb1 = cv::imread( "/home/jin/Data/04_05_2019/sCamSscene/color/1554465502.049380086.png");
    // cv::Mat rgb2 = cv::imread( "/home/jin/Data/04_05_2019/sCamSscene/color/1554465502.719020871.png");
    // cv::Mat depth1 = cv::imread( "/home/jin/Data/04_05_2019/sCamSscene/aligned_depth/1554465502.049380086.png", -1);
    // cv::Mat depth2 = cv::imread( "/home/jin/Data/04_05_2019/sCamSscene/aligned_depth/1554465502.719020871.png", -1);


    // checked // moving camera in feature rich environment
    cv::Mat rgb1 = cv::imread( "/home/jin/Data/04_05_2019/mCamSscene/color/1554465576.399047303.png");
    cv::Mat rgb2 = cv::imread( "/home/jin/Data/04_05_2019/mCamSscene/color/1554465583.298357888.png");
    cv::Mat depth1 = cv::imread( "/home/jin/Data/04_05_2019/mCamSscene/aligned_depth/1554465576.399047303.png", -1);
    cv::Mat depth2 = cv::imread( "/home/jin/Data/04_05_2019/mCamSscene/aligned_depth/1554465583.298357888.png", -1);

    // // Still unsolved 04/05/2019// moving camera in normal environment
    // cv::Mat rgb1 = cv::imread( "/home/jin/Data/04_05_2019/mCamSsceneN/color/1554467649.270468410.png");
    // cv::Mat rgb2 = cv::imread( "/home/jin/Data/04_05_2019/mCamSsceneN/color/1554467651.563677925.png");
    // cv::Mat depth1 = cv::imread( "/home/jin/Data/04_05_2019/mCamSsceneN/aligned_depth/1554467649.270468410.png", -1);
    // cv::Mat depth2 = cv::imread( "/home/jin/Data/04_05_2019/mCamSsceneN/aligned_depth/1554467651.563677925", -1);


    // my moving object intrusion case
    cv::Mat f1c = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/color/1554396069.228104175.png");
    cv::Mat f2c = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/color/1554396070.572613295.png");
    cv::Mat f3c = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/color/1554396071.901310914.png");
    cv::Mat f4c = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/color/1554396073.426762856.png");
    cv::Mat f5c = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/color/1554396075.004447541.png");
    
    cv::Mat f1d = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/aligned_depth/1554396069.228104175.png");
    cv::Mat f2d = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/aligned_depth/1554396070.572613295.png");
    cv::Mat f3d = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/aligned_depth/1554396071.901310914.png");
    cv::Mat f4d = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/aligned_depth/1554396073.426762856.png");
    cv::Mat f5d = cv::imread( "/home/jin/Data/04_04_2019/sCamDscene/aligned_depth/1554396075.004447541.png");

    // cv::Mat rgb1 = f1c;
    // cv::Mat rgb2 = f5c;
    // cv::Mat depth1 = f1d;
    // cv::Mat depth2 = f5d;


    cv::Ptr<cv::FeatureDetector> _detector;
    cv::Ptr<cv::DescriptorExtractor> _descriptor;
    cv::initModule_nonfree();
    _detector = cv::FeatureDetector::create( "GridSIFT" );
    _descriptor = cv::DescriptorExtractor::create( "SIFT" );
    vector< cv::KeyPoint > kp1, kp2; 
    _detector->detect( rgb1, kp1 );
    _detector->detect( rgb2, kp2 );


    cout<<"Key points of two images: "<<kp1.size()<<", "<<kp2.size()<<endl;
    
    // 可视化， 显示关键点
    cv::Mat imgShow;
    cv::drawKeypoints( rgb1, kp1, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    cv::imshow( "keypoints", imgShow );
    cv::imwrite( "./data/keypoints.png", imgShow );
    cv::waitKey(0); //暂停等待一个按键
   
    // 计算描述子
    cv::Mat desp1, desp2;
    _descriptor->compute( rgb1, kp1, desp1 );
    _descriptor->compute( rgb2, kp2, desp2 );

    // 匹配描述子
    vector< cv::DMatch > matches; 
    cv::FlannBasedMatcher matcher;
    matcher.match( desp1, desp2, matches );
    cout<<"Find total "<<matches.size()<<" matches."<<endl;

    // 可视化：显示匹配的特征
    cv::Mat imgMatches;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, matches, imgMatches );
    cv::imshow( "matches", imgMatches );
    cv::imwrite( "./data/matches.png", imgMatches );
    cv::waitKey( 0 );



    // 筛选匹配，把距离太大的去掉
    vector< cv::DMatch > goodMatches;
    double minDis = 9999;
    for ( size_t i=0; i<matches.size(); i++ )
    {
        if ( matches[i].distance < minDis )
            minDis = matches[i].distance;
    }

    for ( size_t i=0; i<matches.size(); i++ )
    {
        cv::Point2f p1 = kp1[matches[i].queryIdx].pt;
        cv::Point2f p2 = kp2[matches[i].trainIdx].pt;

        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (matches[i].distance < 5*minDis && d1>depthL&&d1<depthH && d2>depthL&&d2<depthH ){
            goodMatches.push_back( matches[i] );
            // 将(u,v,d)转成(x,y,z)
            cv::Point3f pt1 ( p1.x, p1.y, d1 );
            cv::Point3f pd1 = point2dTo3d( pt1, C );
            firstPts.push_back( pd1 );

            pts_img.push_back( p2 );

            cv::Point3f pt2 ( p2.x, p2.y, d2 );
            cv::Point3f pd2 = point2dTo3d( pt2, C );
            secondPts.push_back( pd2 );
        }
    }

    // 显示 good matches
    cout<<"good matches="<<goodMatches.size()<<endl;
    cv::drawMatches( rgb1, kp1, rgb2, kp2, goodMatches, imgMatches );
    cv::imshow( "good matches", imgMatches );
    cv::imwrite( "./data/good_matches.png", imgMatches );
    cv::waitKey(0);
    cout << "first visual stage done!"<<endl<<endl;

/*==============================================================================*/

    cv::Mat rvec, tvec, inliers;

    // cout << "firstPts.size() " << firstPts.size()<<endl;
    // cout << "pts_img.size() " << pts_img.size()<<endl;

    // 求解pnp
    cv::solvePnPRansac( firstPts, pts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 5, 100, inliers );

    cout<<"inliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;

    cv::Mat R;
    cv::Rodrigues(rvec, R); // R is 3x3
    cv::Mat T = cv::Mat::eye(4,4,CV_64F);

    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));
    

    cout<<"T="<<T<<endl<<endl;


    // 画出inliers匹配 
    vector< cv::DMatch > matchesShow;
    for (size_t i=0; i<inliers.rows; i++)
    {
        matchesShow.push_back( goodMatches[inliers.ptr<int>(i)[0]] );    
        // cout << inliers.ptr<int>(i)[0]<<endl;
    }

    cv::drawMatches( rgb1, kp1, rgb2, kp2, matchesShow, imgMatches );
    cv::imshow( "inlier matches", imgMatches );
    cv::imwrite( "./data/inliers.png", imgMatches );
    cv::waitKey( 0 );


    double sumDist = 0;
    double sumError = 0;
    int countN = 0;
    int goodinlierN = 0;
    double goodsumError = 0;
    for (size_t i=0; i<inliers.rows; i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p1 = kp1[goodMatches[ inliers.ptr<int>(i)[0] ].queryIdx].pt;
        cv::Point2f p2 = kp2[goodMatches[ inliers.ptr<int>(i)[0] ].trainIdx].pt;

        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (d1 == 0 || d2 == 0){
            continue;
        }

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt1 ( p1.x, p1.y, d1 );
        cv::Point3f pd1 = point2dTo3d( pt1, C );

        cv::Point3f pt2 ( p2.x, p2.y, d2 );
        cv::Point3f pd2 = point2dTo3d( pt2, C );

        // cout << "pd1 "<<pd1 <<endl;
        // cout << "pd2 "<<pd2 <<endl;
        cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);
        cv::Mat dstMat = T*ptMat;
        cv::Point3f projPd1(dstMat.at<double>(0,0), dstMat.at<double>(1,0),dstMat.at<double>(2,0));
        // cout << "projPd1 "<<projPd1 <<endl;

        // cout << "pd1" << pd1 << endl;
        // cout << "pd2" << pd2 << endl;
        // cout << "(pd1*T-pd2) "<< norm(projPd1-pd2)*100 <<"mm"<<endl<<endl;
        // cout << "(pd1-pd2) "<< norm(pd1-pd2)*100 <<"mm" <<endl;

        sumDist = sumDist + norm(pd1-pd2)*100;
        sumError = sumError + norm(projPd1-pd2)*100;
        countN++;
        if ( norm(projPd1-pd2)*100 < 30){
            goodinlierN++;
            goodsumError = goodsumError+ norm(projPd1-pd2)*100;
        }
    }

    cout << "avg error "<< sumError/countN <<"mm" <<endl;
    cout << "avg dist "<< sumDist/countN  <<"mm" <<endl;



    int staticPtsN = 0;
    sumError = 0;

    vector<cv::Point3f> STfirstPts;
    vector< cv::Point2f > STpts_img;
    vector< cv::DMatch > STMatches;
    for (size_t i=0; i<matches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p1 = kp1[matches[ i ].queryIdx].pt;
        cv::Point2f p2 = kp2[matches[ i ].trainIdx].pt;

        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (d1<depthL||d1>depthH || d2<depthL||d2>depthH){
            continue;
        }

        cv::Point3f pt1 ( p1.x, p1.y, d1 );
        cv::Point3f pd1 = point2dTo3d( pt1, C );

        cv::Point3f pt2 ( p2.x, p2.y, d2 );
        cv::Point3f pd2 = point2dTo3d( pt2, C );


        cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);
        cv::Mat dstMat = T*ptMat;
        cv::Point3f projPd1(dstMat.at<double>(0,0), dstMat.at<double>(1,0),dstMat.at<double>(2,0));

        if ( norm(projPd1-pd2)*100 < 30){
            sumError = sumError + norm(projPd1-pd2)*100 ;
            staticPtsN++;    

            STfirstPts.push_back( pd1 );
            STpts_img.push_back( p2 );
            STMatches.push_back( matches[i] );
        }
    }
    cout << "goodinlierN: " << goodinlierN << "  staticPtsN: " << staticPtsN << endl; 
    cout << "static error "<< sumError/staticPtsN <<"mm" <<endl;
    cout << "inlier error "<< goodsumError/goodinlierN <<"mm" <<endl;
    cout << "====================================================" <<endl<<endl;

    cv::solvePnPRansac( STfirstPts, STpts_img, cameraMatrix, cv::Mat(), rvec, tvec, false, 100, 5, 100, inliers );

    cout<<"UpdateInliers: "<<inliers.rows<<endl;
    cout<<"R="<<rvec<<endl;
    cout<<"t="<<tvec<<endl;
    cv::Rodrigues(rvec, R); // R is 3x3
    T = cv::Mat::eye(4,4,CV_64F);

    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    tvec.copyTo(T(cv::Rect(3, 0, 1, 3)));

    cout<<"T="<<T<<endl<<endl;

    sumDist = 0;
    sumError = 0;
    countN = 0;
    goodinlierN = 0;
    goodsumError = 0;
    staticPtsN = 0;
    for (size_t i=0; i<inliers.rows; i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p1 = kp1[STMatches[ inliers.ptr<int>(i)[0] ].queryIdx].pt;
        cv::Point2f p2 = kp2[STMatches[ inliers.ptr<int>(i)[0] ].trainIdx].pt;

        // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (d1 == 0 || d2 == 0){
            continue;
        }

        // 将(u,v,d)转成(x,y,z)
        cv::Point3f pt1 ( p1.x, p1.y, d1 );
        cv::Point3f pd1 = point2dTo3d( pt1, C );

        cv::Point3f pt2 ( p2.x, p2.y, d2 );
        cv::Point3f pd2 = point2dTo3d( pt2, C );

        // cout << "pd1 "<<pd1 <<endl;
        // cout << "pd2 "<<pd2 <<endl;
        cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);
        cv::Mat dstMat = T*ptMat;
        cv::Point3f projPd1(dstMat.at<double>(0,0), dstMat.at<double>(1,0),dstMat.at<double>(2,0));
        // cout << "projPd1 "<<projPd1 <<endl;

        // cout << "pd1" << pd1 << endl;
        // cout << "pd2" << pd2 << endl;
        // cout << "(pd1*T-pd2) "<< norm(projPd1-pd2)*100 <<"mm"<<endl<<endl;
        // cout << "(pd1-pd2) "<< norm(pd1-pd2)*100 <<"mm" <<endl;

        sumDist = sumDist + norm(pd1-pd2)*100;
        sumError = sumError + norm(projPd1-pd2)*100;
        countN++;
        if ( norm(projPd1-pd2)*100 < 30){
            goodinlierN++;
            goodsumError = goodsumError+ norm(projPd1-pd2)*100;
        }
    }

    cout << "avg error "<< sumError/countN <<"mm" <<endl;
    cout << "avg dist "<< sumDist/countN  <<"mm" <<endl;

    for (size_t i=0; i<matches.size(); i++)
    {
        // query 是第一个, train 是第二个
        cv::Point2f p1 = kp1[matches[ i ].queryIdx].pt;
        cv::Point2f p2 = kp2[matches[ i ].trainIdx].pt;

        ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
        ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];
        if (d1<depthL||d1>depthH || d2<depthL||d2>depthH){
            continue;
        }

        cv::Point3f pt1 ( p1.x, p1.y, d1 );
        cv::Point3f pd1 = point2dTo3d( pt1, C );

        cv::Point3f pt2 ( p2.x, p2.y, d2 );
        cv::Point3f pd2 = point2dTo3d( pt2, C );


        cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);
        cv::Mat dstMat = T*ptMat;
        cv::Point3f projPd1(dstMat.at<double>(0,0), dstMat.at<double>(1,0),dstMat.at<double>(2,0));

        if ( norm(projPd1-pd2)*100 < 30){
            sumError = sumError + norm(projPd1-pd2)*100 ;
            staticPtsN++;    
        }
    }
    cout << "goodinlierN: " << goodinlierN << "  staticPtsN: " << staticPtsN << endl; 
    cout << "static error "<< sumError/staticPtsN <<"mm" <<endl;
    cout << "inlier error "<< goodsumError/goodinlierN <<"mm" <<endl;
    cout << "====================================================" <<endl<<endl;

/*=======================================3D-3D method====================================================*/

    // cout<<"firstPts: "<<firstPts.size()<<endl;
    // cv::Mat outM3by4;// = cv::Mat::zeros(3,4,CV_64F);
    // cv::Mat inliers3d;
    // // moving points from first frame to second frame
    // cv::estimateAffine3D(secondPts,firstPts, outM3by4, inliers3d, 3, 0.999); 
    // cv::Mat rmat = outM3by4(cv::Rect(0,0,3,3));
    // cv::Mat rvecN;

    // cv::Rodrigues(rmat,rvecN);

    // cv::Mat tvecN = outM3by4(cv::Rect(3,0,1,3));


    // // cout<<"affine M = "<<outM3by4<<endl;
    // cout<<"R="<<rvecN<<endl;
    // cout<<"t="<<tvecN<<endl;
    // cout<<"inliers3d: "<<inliers3d.rows<<endl;

    // T = cv::Mat::eye(4,4,CV_64F);
    // rmat.copyTo(T(cv::Rect(0, 0, 3, 3)));
    // rvecN.copyTo(T(cv::Rect(3, 0, 1, 3)));
    

    // cout<<"T="<<T<<endl;

    // sumDist = 0;
    // sumError = 0;
    // countN = 0;
    // for (size_t i=0; i<inliers.rows; i++)
    // {
    //     // query 是第一个, train 是第二个
    //     cv::Point2f p1 = kp1[goodMatches[ inliers.ptr<int>(i)[0] ].queryIdx].pt;
    //     cv::Point2f p2 = kp2[goodMatches[ inliers.ptr<int>(i)[0] ].trainIdx].pt;

    //     // 获取d是要小心！x是向右的，y是向下的，所以y才是行，x是列！
    //     ushort d1 = depth1.ptr<ushort>( int(p1.y) )[ int(p1.x) ];
    //     ushort d2 = depth2.ptr<ushort>( int(p2.y) )[ int(p2.x) ];

    //     // 将(u,v,d)转成(x,y,z)
    //     cv::Point3f pt1 ( p1.x, p1.y, d1 );
    //     cv::Point3f pd1 = point2dTo3d( pt1, C );
    //     pts_inlier1.push_back( pd1 );

    //     cv::Point3f pt2 ( p2.x, p2.y, d2 );
    //     cv::Point3f pd2 = point2dTo3d( pt2, C );
    //     pts_inlier2.push_back( pd2 );

    //     // cout << "pd1 "<<pd1 <<endl;
    //     // cout << "pd2 "<<pd2 <<endl;
    //     cv::Mat ptMat = (cv::Mat_<double>(4, 1) << pd1.x, pd1.y, pd1.z, 1);
    //     cv::Mat dstMat = T*ptMat;
    //     cv::Point3f projPd1(dstMat.at<double>(0,0), dstMat.at<double>(1,0),dstMat.at<double>(2,0));
    //     // cout << "projPd1 "<<projPd1 <<endl;

    //     cout << "(pd1*T-pd2) "<< norm(projPd1-pd2)*100 <<"mm" <<endl;
    //     // cout << "(pd1-pd2) "<< norm(pd1-pd2)*100 <<"mm" <<endl;

    //     sumDist = sumDist + norm(pd1-pd2)*100;
    //     sumError = sumError + norm(projPd1-pd2)*100;
    //     countN++;

    // }

    // cout << "avg error "<< sumError/countN <<"mm" <<endl;
    // cout << "avg dist "<< sumDist/countN <<"mm" <<endl;




    return 0;
}
