#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <limits>

namespace py = pybind11;

// Configuration
const int NUM_ENEMIES_DEFAULT = 3;
const double MAP_SIZE_DEFAULT = 1.0;
const double HIT_RADIUS_DEFAULT = 0.05;
const int MAX_STEPS = 50;
const double DAMAGE_PER_TURN = 0.5; // Increased damage to make the strategy obvious

struct Enemy
{
    double x, y;
    bool alive;
};

struct StepResult
{
    std::vector<double> observation;
    double reward;
    bool done;
    std::string info;
};

class HybridJamShoot
{
private:
    std::vector<Enemy> enemies;
    int steps;
    bool independent_mode;
    int num_enemies;
    double map_size;
    double hit_radius;
    std::mt19937 rng;

    double dist(double x1, double y1, double x2, double y2)
    {
        return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
    }

public:
    HybridJamShoot(bool mode = false,
                   int n_enemies = NUM_ENEMIES_DEFAULT,
                   double size = MAP_SIZE_DEFAULT,
                   double radius = HIT_RADIUS_DEFAULT)
        : independent_mode(mode),
          num_enemies(n_enemies),
          map_size(size),
          hit_radius(radius),
          rng(std::random_device{}()) {}

    std::vector<double> reset()
    {
        enemies.clear();
        steps = 0;
        std::uniform_real_distribution<double> dist_pos(0.0, map_size);
        for (int i = 0; i < num_enemies; ++i)
        {
            enemies.push_back({dist_pos(rng), dist_pos(rng), true});
        }
        return get_observation();
    }

    StepResult step(int discrete_act, std::vector<double> continuous_act)
    {
        steps++;
        double reward = -0.1; // Base time penalty
        bool done = false;
        std::string info = "";

        if (continuous_act.size() != 2)
            return {get_observation(), -1.0, true, "Invalid action"};

        double shot_x = continuous_act[0];
        double shot_y = continuous_act[1];

        bool valid_jam = (discrete_act >= 0 && discrete_act < enemies.size());

        // ==========================================================
        // PHASE 1: ENEMY OFFENSE (Damage Calculation)
        // ==========================================================
        // This MUST happen before we process the player's shot.
        // Even if an enemy is about to die in Phase 2, they are
        // currently "Alive" and will deal damage unless jammed.

        if (independent_mode)
        {
            for (int i = 0; i < enemies.size(); ++i)
            {
                // If enemy is alive...
                if (enemies[i].alive)
                {
                    // And NOT jammed...
                    if (!valid_jam || i != discrete_act)
                    {
                        // They deal damage!
                        reward -= DAMAGE_PER_TURN;
                    }
                }
            }
        }

        // ==========================================================
        // PHASE 2: PLAYER OFFENSE (Resolution)
        // ==========================================================
        // Now we calculate if the player's shot kills anyone.

        if (independent_mode)
        {
            // Check shot against ALL enemies
            int best_target = -1;
            double closest_dist = std::numeric_limits<double>::max();

            for (int i = 0; i < enemies.size(); ++i)
            {
                if (enemies[i].alive)
                {
                    double d = dist(shot_x, shot_y, enemies[i].x, enemies[i].y);
                    if (d <= hit_radius && d < closest_dist)
                    {
                        closest_dist = d;
                        best_target = i;
                    }
                }
            }

            // Apply Death
            if (best_target != -1)
            {
                enemies[best_target].alive = false;
                reward += 10.0;
                info = "Hit Enemy " + std::to_string(best_target);
            }
        }
        else
        {
            // Dependent Mode Logic (unchanged)
            if (!valid_jam)
                reward -= 1.0;
            else if (!enemies[discrete_act].alive)
                reward -= 0.5;
            else
            {
                Enemy &target = enemies[discrete_act];
                if (dist(shot_x, shot_y, target.x, target.y) <= hit_radius)
                {
                    target.alive = false;
                    reward += 10.0;
                    info = "Hit Jammed Target";
                }
                else
                {
                    reward -= 0.2;
                }
            }
        }

        // Check Termination
        bool all_dead = true;
        for (const auto &e : enemies)
            if (e.alive)
                all_dead = false;

        if (all_dead)
        {
            reward += 5.0;
            done = true;
        }
        else if (steps >= MAX_STEPS)
        {
            done = true;
        }

        return {get_observation(), reward, done, info};
    }

    std::vector<double> get_observation()
    {
        std::vector<double> obs;
        for (const auto &e : enemies)
        {
            obs.push_back(e.x);
            obs.push_back(e.y);
            obs.push_back(e.alive ? 1.0 : 0.0);
        }
        return obs;
    }

    int get_num_enemies() { return num_enemies; }
};

PYBIND11_MODULE(_hybrid_shoot, m)
{
    py::class_<StepResult>(m, "StepResult")
        .def_readwrite("observation", &StepResult::observation)
        .def_readwrite("reward", &StepResult::reward)
        .def_readwrite("done", &StepResult::done)
        .def_readwrite("info", &StepResult::info);

    py::class_<HybridJamShoot>(m, "HybridJamShoot")
        .def(py::init<bool, int, double, double>(),
             py::arg("independent_mode") = false,
             py::arg("n_enemies") = 3,
             py::arg("map_size") = 1.0,
             py::arg("hit_radius") = 0.05)
        .def("reset", &HybridJamShoot::reset)
        .def("step", &HybridJamShoot::step)
        .def("get_num_enemies", &HybridJamShoot::get_num_enemies);
}