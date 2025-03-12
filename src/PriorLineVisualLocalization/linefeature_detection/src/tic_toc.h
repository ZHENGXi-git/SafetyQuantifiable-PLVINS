//
// Created by zx on 22年4月27日.
//

#ifndef INC_2D_3D_POSE_TRACKING_MASTER_TIC_TOC_H
#define INC_2D_3D_POSE_TRACKING_MASTER_TIC_TOC_H

#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc
{
public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};


#endif //INC_2D_3D_POSE_TRACKING_MASTER_TIC_TOC_H
