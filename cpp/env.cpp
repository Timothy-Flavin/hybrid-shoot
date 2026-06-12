#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <limits>
#include <utility>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

// Configuration
const int NUM_ENEMIES_DEFAULT = 3;
const double MAP_SIZE_DEFAULT = 1.0;
const double HIT_RADIUS_DEFAULT = 0.05;
const int MAX_STEPS = 50;
const double DAMAGE_PER_TURN = 0.5;

struct Enemy
{
    double x, y;
    bool alive;
};

// ==========================================================================
// Core simulation. Pure C++, no Python types, so it can be stepped from
// inside an OpenMP parallel region with the GIL released. Both the single
// pybind class and the vectorized class wrap one or many of these.
// ==========================================================================
struct Core
{
    std::vector<Enemy> enemies;
    int steps = 0;
    bool independent_mode;
    int num_enemies;
    double map_size;
    double hit_radius;
    bool joint_xy_action;
    int hilbert_width;
    std::mt19937 rng;

    Core(bool mode, int n_enemies, double size, double radius,
         bool joint, int hwidth, uint32_t seed)
        : independent_mode(mode), num_enemies(n_enemies), map_size(size),
          hit_radius(radius), joint_xy_action(joint), hilbert_width(hwidth),
          rng(seed)
    {
        enemies.resize(num_enemies);
    }

    int obs_dim() const { return num_enemies * 3; }
    int cont_dim() const { return joint_xy_action ? 1 : 2; }

    static double dist(double x1, double y1, double x2, double y2)
    {
        double dx = x1 - x2, dy = y1 - y2;
        return std::sqrt(dx * dx + dy * dy);
    }

    // Standard Hilbert d -> (x, y) on an n x n grid (n a power of two).
    static void d2xy(int n, int d, int &x, int &y)
    {
        int rx, ry, t = d;
        x = 0;
        y = 0;
        for (int s = 1; s < n; s *= 2)
        {
            rx = 1 & (t / 2);
            ry = 1 & (t ^ rx);
            if (ry == 0)
            {
                if (rx == 1)
                {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                int tmp = x;
                x = y;
                y = tmp;
            }
            x += s * rx;
            y += s * ry;
            t /= 4;
        }
    }

    // Map a scalar in [0, 1] to a point covering the full [0, map_size]^2 map.
    // Dividing by (n - 1) and scaling by map_size makes the grid span the whole
    // map (the old wrapper divided by n, leaving a 1/n strip unreachable), and
    // clamping d removes the endpoint aliasing where scalar==1 wrapped to (0,0).
    std::pair<double, double> scalar_to_xy(double scalar) const
    {
        int n = hilbert_width;
        long d = std::lround(scalar * (static_cast<double>(n) * n));
        if (d < 0)
            d = 0;
        if (d > static_cast<long>(n) * n - 1)
            d = static_cast<long>(n) * n - 1;
        int x, y;
        d2xy(n, static_cast<int>(d), x, y);
        double denom = (n > 1) ? (n - 1) : 1;
        return {map_size * x / denom, map_size * y / denom};
    }

    void reset()
    {
        steps = 0;
        std::uniform_real_distribution<double> dist_pos(0.0, map_size);
        for (int i = 0; i < num_enemies; ++i)
            enemies[i] = {dist_pos(rng), dist_pos(rng), true};
    }

    void write_obs(double *out) const
    {
        for (int i = 0; i < num_enemies; ++i)
        {
            out[i * 3 + 0] = enemies[i].x;
            out[i * 3 + 1] = enemies[i].y;
            out[i * 3 + 2] = enemies[i].alive ? 1.0 : 0.0;
        }
    }

    // Advance one step. `cont` points to cont_dim() doubles. Returns reward
    // and sets `terminated` (episode reached a true terminal state: all enemies
    // dead) and `truncated` (hit the step limit with enemies still alive) as
    // separate flags, so a learner can bootstrap correctly on truncation.
    double step(int discrete_act, const double *cont, int cont_len,
                bool &terminated, bool &truncated, std::string *info = nullptr)
    {
        steps++;
        double reward = -0.1; // Base time penalty
        terminated = false;
        truncated = false;

        double shot_x, shot_y;
        if (joint_xy_action)
        {
            if (cont_len < 1)
            {
                terminated = true;
                return -1.0;
            }
            auto xy = scalar_to_xy(cont[0]);
            shot_x = xy.first;
            shot_y = xy.second;
        }
        else
        {
            if (cont_len < 2)
            {
                terminated = true;
                return -1.0;
            }
            shot_x = cont[0];
            shot_y = cont[1];
        }

        bool valid_jam = (discrete_act >= 0 && discrete_act < num_enemies);

        // ---- PHASE 1: enemy offense (damage happens before the player's shot)
        if (independent_mode)
        {
            for (int i = 0; i < num_enemies; ++i)
            {
                if (enemies[i].alive && (!valid_jam || i != discrete_act))
                    reward -= DAMAGE_PER_TURN;
            }
        }

        // ---- PHASE 2: player offense (resolve the shot)
        if (independent_mode)
        {
            int best_target = -1;
            double closest_dist = std::numeric_limits<double>::max();
            for (int i = 0; i < num_enemies; ++i)
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
            if (best_target != -1)
            {
                enemies[best_target].alive = false;
                reward += 10.0;
                if (info)
                    *info = "Hit Enemy " + std::to_string(best_target);
            }
        }
        else
        {
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
                    if (info)
                        *info = "Hit Jammed Target";
                }
                else
                    reward -= 0.2;
            }
        }

        // ---- Termination vs. truncation
        bool all_dead = true;
        for (const auto &e : enemies)
            if (e.alive)
                all_dead = false;

        if (all_dead)
        {
            reward += 5.0;
            terminated = true; // true terminal: bootstrap value is 0
        }
        else if (steps >= MAX_STEPS)
            truncated = true; // time limit: still bootstrap from final obs

        return reward;
    }
};

struct StepResult
{
    py::array_t<double> observation;
    double reward;
    bool terminated;
    bool truncated;
    bool done; // convenience: terminated || truncated
    std::string info;
};

// ==========================================================================
// Single environment. Returns the observation as a numpy array directly so
// the Python wrapper does not have to rebuild it with np.array().
// ==========================================================================
class HybridJamShoot
{
private:
    Core core;

public:
    HybridJamShoot(bool mode = false,
                   int n_enemies = NUM_ENEMIES_DEFAULT,
                   double size = MAP_SIZE_DEFAULT,
                   double radius = HIT_RADIUS_DEFAULT,
                   bool joint_xy_action = false,
                   int hilbert_width = 16)
        : core(mode, n_enemies, size, radius, joint_xy_action, hilbert_width,
               std::random_device{}()) {}

    py::array_t<double> make_obs()
    {
        py::array_t<double> obs(core.obs_dim());
        core.write_obs(obs.mutable_data());
        return obs;
    }

    py::array_t<double> reset()
    {
        core.reset();
        return make_obs();
    }

    StepResult step(int discrete_act,
                    py::array_t<double, py::array::c_style | py::array::forcecast> cont)
    {
        StepResult r;
        r.reward = core.step(discrete_act, cont.data(),
                             static_cast<int>(cont.size()),
                             r.terminated, r.truncated, &r.info);
        r.done = r.terminated || r.truncated;
        r.observation = make_obs();
        return r;
    }

    std::pair<double, double> scalar_to_xy(double s) { return core.scalar_to_xy(s); }
    int get_num_enemies() { return core.num_enemies; }
    int get_obs_dim() { return core.obs_dim(); }
};

// ==========================================================================
// Vectorized environment. Holds num_envs independent cores and steps them in
// parallel with OpenMP, keeping all the multiprocessing inside C++. The API
// (batched reset/step with autoreset) mirrors Gym's VectorEnv / envpool.
// ==========================================================================
class VecHybridJamShoot
{
private:
    std::vector<Core> cores;
    int num_envs;
    int obs_dim;
    int cont_dim;

public:
    VecHybridJamShoot(int n_envs,
                      bool mode = false,
                      int n_enemies = NUM_ENEMIES_DEFAULT,
                      double size = MAP_SIZE_DEFAULT,
                      double radius = HIT_RADIUS_DEFAULT,
                      bool joint_xy_action = false,
                      int hilbert_width = 16,
                      int num_threads = 0)
        : num_envs(n_envs)
    {
        std::random_device rd;
        cores.reserve(n_envs);
        for (int e = 0; e < n_envs; ++e)
            cores.emplace_back(mode, n_enemies, size, radius,
                               joint_xy_action, hilbert_width, rd());
        obs_dim = cores[0].obs_dim();
        cont_dim = cores[0].cont_dim();
#ifdef _OPENMP
        if (num_threads > 0)
            omp_set_num_threads(num_threads);
#endif
    }

    py::array_t<double> reset()
    {
        py::array_t<double> obs({num_envs, obs_dim});
        double *op = obs.mutable_data();
        {
            py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
            for (int e = 0; e < num_envs; ++e)
            {
                cores[e].reset();
                cores[e].write_obs(op + static_cast<size_t>(e) * obs_dim);
            }
        }
        return obs;
    }

    // actions: discrete (num_envs,), continuous (num_envs, cont_dim).
    // Returns (obs, rewards, terminated, truncated, final_obs):
    //   obs        -- next observation; for an env that just ended this is the
    //                 first observation of the freshly-reset episode (autoreset)
    //   terminated -- reached a true terminal state (bootstrap target = 0)
    //   truncated  -- hit the step limit with enemies alive (still bootstrap)
    //   final_obs  -- the real post-step observation that actually occurred
    //                 (the s' to bootstrap from); for envs that did NOT end it
    //                 equals obs. Only meaningful where terminated | truncated.
    py::tuple step(
        py::array_t<int, py::array::c_style | py::array::forcecast> disc_arr,
        py::array_t<double, py::array::c_style | py::array::forcecast> cont_arr)
    {
        const int *disc = disc_arr.data();
        const double *cont = cont_arr.data();

        py::array_t<double> obs({num_envs, obs_dim});
        py::array_t<double> final_obs({num_envs, obs_dim});
        py::array_t<double> rewards(num_envs);
        py::array_t<bool> terminated(num_envs);
        py::array_t<bool> truncated(num_envs);
        double *op = obs.mutable_data();
        double *fp = final_obs.mutable_data();
        double *rp = rewards.mutable_data();
        bool *tp = terminated.mutable_data();
        bool *up = truncated.mutable_data();

        {
            py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
            for (int e = 0; e < num_envs; ++e)
            {
                bool term, trunc;
                double r = cores[e].step(
                    disc[e], cont + static_cast<size_t>(e) * cont_dim, cont_dim,
                    term, trunc);

                double *fobs = fp + static_cast<size_t>(e) * obs_dim;
                double *mobs = op + static_cast<size_t>(e) * obs_dim;

                // The genuine observation that resulted from this step (s').
                cores[e].write_obs(fobs);

                if (term || trunc)
                {
                    // Autoreset: the returned obs is the new episode's first
                    // observation; the true s' is preserved in final_obs.
                    cores[e].reset();
                    cores[e].write_obs(mobs);
                }
                else
                {
                    std::copy(fobs, fobs + obs_dim, mobs);
                }

                rp[e] = r;
                tp[e] = term;
                up[e] = trunc;
            }
        }
        return py::make_tuple(obs, rewards, terminated, truncated, final_obs);
    }

    int get_num_envs() { return num_envs; }
    int get_num_enemies() { return cores[0].num_enemies; }
    int get_obs_dim() { return obs_dim; }
    int get_cont_dim() { return cont_dim; }
};

PYBIND11_MODULE(_hybrid_shoot, m)
{
    py::class_<StepResult>(m, "StepResult")
        .def_readwrite("observation", &StepResult::observation)
        .def_readwrite("reward", &StepResult::reward)
        .def_readwrite("terminated", &StepResult::terminated)
        .def_readwrite("truncated", &StepResult::truncated)
        .def_readwrite("done", &StepResult::done)
        .def_readwrite("info", &StepResult::info);

    py::class_<HybridJamShoot>(m, "HybridJamShoot")
        .def(py::init<bool, int, double, double, bool, int>(),
             py::arg("independent_mode") = false,
             py::arg("n_enemies") = 3,
             py::arg("map_size") = 1.0,
             py::arg("hit_radius") = 0.05,
             py::arg("joint_xy_action") = false,
             py::arg("hilbert_width") = 16)
        .def("reset", &HybridJamShoot::reset)
        .def("step", &HybridJamShoot::step,
             py::arg("discrete_act"), py::arg("continuous_act"))
        .def("scalar_to_xy", &HybridJamShoot::scalar_to_xy)
        .def("get_num_enemies", &HybridJamShoot::get_num_enemies)
        .def("get_obs_dim", &HybridJamShoot::get_obs_dim);

    py::class_<VecHybridJamShoot>(m, "VecHybridJamShoot")
        .def(py::init<int, bool, int, double, double, bool, int, int>(),
             py::arg("num_envs"),
             py::arg("independent_mode") = false,
             py::arg("n_enemies") = 3,
             py::arg("map_size") = 1.0,
             py::arg("hit_radius") = 0.05,
             py::arg("joint_xy_action") = false,
             py::arg("hilbert_width") = 16,
             py::arg("num_threads") = 0)
        .def("reset", &VecHybridJamShoot::reset)
        .def("step", &VecHybridJamShoot::step,
             py::arg("discrete_acts"), py::arg("continuous_acts"))
        .def("get_num_envs", &VecHybridJamShoot::get_num_envs)
        .def("get_num_enemies", &VecHybridJamShoot::get_num_enemies)
        .def("get_obs_dim", &VecHybridJamShoot::get_obs_dim)
        .def("get_cont_dim", &VecHybridJamShoot::get_cont_dim);
}
