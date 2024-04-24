#pragma once

#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
namespace py = pybind11;

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 800;

//taregt 
int min_y = 25;
int max_y = 25; //25 + 25 so 50
int min_speed = 5;
int max_speed = 15; //5 + 15 = 20
int min_radius = 20;
int max_radius = 80;
//proj
float projectile_speed{ 45.f };
//player
int player_max_cooldown{ 45 };
float player_radius{ 50.f };
sf::Vector2f player_pos{ SCREEN_WIDTH / 2, SCREEN_HEIGHT - 100 };

//game

int EPISODE_MAX_STEPS = 10000;


// Function to calculate the direction vector normalized
sf::Vector2f calculate_direction(sf::Vector2f start, sf::Vector2f end) {
    sf::Vector2f direction = end - start;
    float length = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    return direction / length;
}

float calculate_distance(sf::Vector2f a, sf::Vector2f b) {
    sf::Vector2f delta = b - a;
    return std::sqrt(delta.x * delta.x + delta.y * delta.y);
}

class Target
{
       
    sf::Color color{ sf::Color::Red };

    sf::CircleShape shape;
    sf::Vector2f position;
    sf::Vector2f speed;
    int radius; 

public:

    Target() {
        respawn();
    }

    void update() {
        position += speed;
        position.x -= position.x > SCREEN_WIDTH ? SCREEN_WIDTH : 0;
        shape.setPosition(position);
    }
    void draw(sf::RenderWindow& window) const {
        window.draw(shape);
    }

    void respawn() {
        position = sf::Vector2f(rand() % SCREEN_WIDTH, rand() % max_y + min_y);
        speed = sf::Vector2f(rand() % max_speed + min_speed, 0);
        radius = rand() % max_radius + min_radius;
        shape = sf::CircleShape(radius);
        shape.setFillColor(color);
        shape.setPosition(position);
    }


    auto get_center() const {
        return position + sf::Vector2f{ float(radius), float(radius)};
    }
    auto get_radius() const {
        return radius;
    }
    auto get_speed() const {
        return speed.x;
    }

};

class Projectile
{

    sf::CircleShape shape{radius};
    sf::Color color{ sf::Color::White };

    sf::Vector2f position;
    sf::Vector2f direction;

public:
    static constexpr float radius{ 5.f };

    Projectile(sf::Vector2f position, sf::Vector2f direction) : position{position}, direction{ direction }
    {
    }

    void update() {
        position += direction * projectile_speed;
        shape.setPosition(position);
    }
    void draw(sf::RenderWindow& window) const {
        window.draw(shape);
    }
    bool dead() const {
        return position.x < 0 or position.x > SCREEN_WIDTH or position.y < 0 or position.y > SCREEN_HEIGHT;
    }
    auto get_center() const {
        return position + sf::Vector2f{ radius, radius };
    }

    bool hit(const Target& target) {
        return calculate_distance(position, target.get_center()) < (radius + target.get_radius());
    }
};

class Player
{

    sf::Color color{ sf::Color::Cyan }; 
    sf::CircleShape shape{ player_radius };
    sf::Vector2f position{ player_pos };

    int cooldown{ 0 };

public:

    Player() {
        shape.setPosition(position);
        shape.setFillColor(color);
    }

    auto get_center() const {
        return position + sf::Vector2f{ player_radius, player_radius };
    }
    void update() {
        --cooldown;
    }

    void draw(sf::RenderWindow& window) const {
        window.draw(shape);
    }

    bool shoot() {
        
        if (cooldown > 0)
            return false;

        cooldown = player_max_cooldown;
        return true;
    }
};
struct Input{
    bool shoot;
    float x;
    float y;
};

struct Action {
    bool shoot;
    sf::Vector2f shot_direction;
    Action(bool shoot, sf::Vector2f dir) : shoot{ shoot }, shot_direction{ dir }
    {
    }
};

class ShootingGame
{

    Target target{};
    Player player{};
    std::vector<Projectile> projectiles{};
    sf::RenderWindow* window{};
    
    int current_step{0};

    int update()
    {
        int reward = 0;

        target.update();
        for (int i = 0; i < projectiles.size(); ++i)
        {
            auto& p = projectiles[i];
            p.update();
            bool hit = p.hit(target);
            if (hit) {
                reward = 1;
                target.respawn();
            }
            if (hit or p.dead()) {
                projectiles.erase(projectiles.begin() + i);
            }
        }
        player.update();
        return reward;
    }

    auto make_obs() {
        return py::make_tuple( target.get_center().x, target.get_center().y, target.get_speed(), target.get_radius() );
    }

    bool make_info()
    {
        return false;
    }

public:

    ShootingGame() {

    }

    auto get_window()
    {
        return window;
    }

    void init_render(){
        window = new sf::RenderWindow{sf::VideoMode(SCREEN_WIDTH, SCREEN_HEIGHT), "SFML Target Shooting Game"};
        window ->setFramerateLimit(60);
    }

    auto reset()
    {
        return py::make_tuple(make_obs(), make_info());
    }

    auto step(py::tuple input)
    {
        bool shoot = input[0].cast<bool>();
        float x = input[1].cast<float>();
        float y = input[2].cast<float>();

        Action action {shoot, {x, y}};

        if (action.shoot && player.shoot()) {
            projectiles.push_back(Projectile(player.get_center(), calculate_direction(player.get_center(), action.shot_direction)));
        }
        int reward = update();
        ++current_step;
        bool trunc = current_step >= EPISODE_MAX_STEPS;

        return py::make_tuple (make_obs(), reward, false, trunc, make_info() );
    }

    void draw() const
    {
        window->clear();

        player.draw(*window);
        target.draw(*window);

        for (auto& p : projectiles)
            p.draw(*window);

        window->display();
    }
};


class HumanRenderGame
{
public:
    HumanRenderGame()
    {
    }

    void play()
    {
        auto game = ShootingGame();
        game.init_render();
        auto window = game.get_window();
        while (window->isOpen()) {
            sf::Event event;
            bool shoot = false;
            int x = 0;
            int y = 0;
        
            while (window->pollEvent(event)) {
                if (event.type == sf::Event::Closed) {
                    window->close();
                }

                // Check for mouse click to shoot a projectile
                if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                    //sf::Vector2f mousePos(rand() % SCREEN_WIDTH, 35);
                    shoot = true;
                    x = event.mouseButton.x;
                    y = event.mouseButton.y;
                }
            }

            int reward = game.step(py::make_tuple(shoot, x, y))[1].cast<int>();
            if (reward)
                cout << "reward " << reward <<endl;
            game.draw();
        }

    }
};

