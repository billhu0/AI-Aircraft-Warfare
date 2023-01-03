import sys
import random
import time
import traceback
import pygame
import pygame.locals
from typing import *  # for typing alias only

# Game framework 游戏框架
import myplane
import enemy
import bullet
import supply

# Q-learning, DQN 
import q_learning
import state

bullets = []

def print_green(s: str, end: str or None = '\n') -> None:
    print("\033[37;42m{}\033[0m ".format(s), end=end)

def print_red(s: str, end: str or None = '\n') -> None:
    print("\033[37;41m{}\033[0m ".format(s), end=end)

"""
这是重构过的游戏的类 class PlaneWar, 其中:
    start() 方法是手动键盘控制游戏
    train() 方法是q-learning训练
    这两个方法里就是游戏的主要逻辑, 主要包含一个init操作和一个while True循环
        init_game() 中包含了一些 self.什么什么的成员变量，可以从这些变量获取游戏状态信息
        那个while循环就是游戏的主循环, 我们主要需要关注这个循环, 每一次循环都要更新state并进行相关操作
    move_me() 是手动键盘控制飞机的移动, 在手动控制游戏时会用到
    move_action() 是根据action移动飞机, 在q-learning训练时会用到
    其余的私有方法(用_开头的函数)是游戏框架原有的代码，可以不看
"""
class PlaneWar:

    def __init__(self) -> None:
        """Init pygame and load images and sound effects."""

        # Pygame initialization
        pygame.init()
        pygame.mixer.init()
        pygame.display.set_caption("Plane Warfare -- 飞机大战")

        # 游戏界面初始化 Game interface initialization
        self.width, self.height = 480, 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.background = pygame.image.load('images/background.png').convert()

        # 加载背景音乐和游戏音效 Preload music and sound effects
        self.sounds: Dict[str, pygame.mixer.Sound] = {
            'bullet': pygame.mixer.Sound('sound/bullet.wav'),
            'enemy1_down': pygame.mixer.Sound('sound/enemy1_down.wav'),
            'enemy2_down': pygame.mixer.Sound('sound/enemy2_down.wav'),
            'enemy3_down': pygame.mixer.Sound('sound/enemy3_down.wav'),
            'enemy3_flying': pygame.mixer.Sound('sound/enemy3_flying.wav'),
            'bomb': pygame.mixer.Sound('sound/use_bomb.wav'),
            'get_bomb': pygame.mixer.Sound('sound/get_bomb.wav'),
            'get_bullet': pygame.mixer.Sound('sound/get_bullet.wav'),
            'upgrade': pygame.mixer.Sound('sound/upgrade.wav'),
            'supply': pygame.mixer.Sound('sound/supply.wav'),
            'me_down': pygame.mixer.Sound('sound/me_down.wav'),
        }
        [sound.set_volume(0.2) for sound in self.sounds.values()]

        pygame.mixer.music.load('sound/game_music.ogg')
        pygame.mixer.music.set_volume(0.2)

        # 加载图像资源 Preload images
        self.pause_nor_image = pygame.image.load('images/pause_nor.png').convert_alpha()
        self.pause_pressed_image = pygame.image.load('images/pause_pressed.png').convert_alpha()
        self.resume_nor_image = pygame.image.load('images/resume_nor.png').convert_alpha()
        self.resume_pressed_image = pygame.image.load('images/resume_pressed.png').convert_alpha()
        self.bomb_image = pygame.image.load('images/bomb.png').convert_alpha()  # 全屏炸弹
        self.life_image = pygame.image.load('images/life.png').convert_alpha()  # 生命数量
        self.again_image = pygame.image.load('images/again.png').convert_alpha()  # game over
        self.gameover_image = pygame.image.load('images/gameover.png').convert_alpha()  # game over

        # 加载字体 Preload fonts
        self.score_font = pygame.font.Font('font/font.ttf', 36)
        self.bomb_font = pygame.font.Font('font/font.ttf', 48)
        self.gameover_font = pygame.font.Font('font/font.ttf', 48)

    def init_game(self) -> None:
        """
        This function is called at the beginning of a game.
        It generates enemy aircraft, bullets, as well as initializes timers and game status information.
        在一局游戏开始前会调用这个函数.
        这个函数会生成我方飞机、敌机，以及初始化计时器和游戏状态信息.
        (可以观察这个method中出现的变量, 游戏状态信息可以这么获取.
        """

        # Generate enemy aircraft and my aircraft 生成敌机和我方飞机
        self.me = myplane.MyPlane((self.width, self.height))  # 我方飞机

        self.enemies = pygame.sprite.Group()  # 敌方飞机(所有飞机)
        self.small_enemies = pygame.sprite.Group()  # 敌方小型飞机
        self.mid_enemies = pygame.sprite.Group()  # 敌方中型飞机
        self.big_enemies = pygame.sprite.Group()  # 敌方大型飞机

        self._add_small_enemies(15)
        self._add_mid_enemies(4)
        self._add_big_enemies(2)

        # 生成普通子弹 Generate normal bullets
        self.bullet_1 = [bullet.Bullet1(self.me.rect.midtop) for _ in range(4)]
        self.bullet_1_index = 0

        # 生成超级子弹 Generate super bullets
        self.bullet_2 = [bullet.Bullet2((self.me.rect.centerx - 33, self.me.rect.centery)) for _ in range(8)]
        self.bullet_2_index = 0

        # 中弹图片索引
        self.e1_destroy_index = 0
        self.e2_destroy_index = 0
        self.e3_destroy_index = 0
        self.me_destroy_index = 0

        # Miscellaneous
        self.paused_rect = self.pause_nor_image.get_rect()
        self.paused_rect.left, self.paused_rect.top = self.width - self.paused_rect.width - 10, 10
        self.paused_image = self.pause_nor_image
        self.bomb_rect = self.bomb_image.get_rect()

        # 每30秒发放一个补给包
        self.bullet_supply = supply.Bullet_Supply((self.width, self.height))
        self.bomb_supply = supply.Bomb_Supply((self.width, self.height))
        self.SUPPLY_TIME = pygame.USEREVENT
        pygame.time.set_timer(self.SUPPLY_TIME, 30 * 1000)

        # 超级子弹定时器
        self.DOUBLE_BULLET_TIME = pygame.USEREVENT + 1

        # 解除我方无敌状态定时器
        self.INVINCIBLE_TIME = pygame.USEREVENT + 2

        # 生命数量
        self.life_rect = self.life_image.get_rect()

        # 用于阻止重复打开记录文件
        self.recorded = False

        # Game over interface
        self.again_rect = self.again_image.get_rect()
        self.gameover_rect = self.gameover_image.get_rect()

        # 用于切换图片
        self.switch_image = True
        self.delay = 100

        # Initialize game status 初始化游戏状态信息
        self.score: int = 0  # 分数
        self.life_num: int = 3  # 剩余命的数量
        self.paused: bool = False  # 是否处于暂停状态
        self.level: int = 1  # 难度等级
        self.bomb_num: int = 3  # 炸弹(可清空全屏飞机)数量
        self.is_double_bullet: bool = False  # 是否处于超级子弹模式(一次发左右2颗子弹)
        self.clock = pygame.time.Clock()
        pygame.mixer.music.play(-1)

    def start(self):
        # Initialize game
        self.init_game()

        # Main loop 主循环, 循环一次画一帧
        while True:
            self._handle_events()
            self._add_difficulty()

            self.print_info()

            if not self.paused and self.life_num:
                self.move_me()
                self._draw_frame()
            elif self.life_num == 0:
                self._draw_game_over()
                # 检测用户的鼠标操作
                # 如果用户按下鼠标左键
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    # 如果用户点击“重新开始”
                    if self.again_rect.left < pos[0] < self.again_rect.right and \
                            self.again_rect.top < pos[1] < self.again_rect.bottom:
                        # 调用main函数，重新开始游戏
                        self.start()
                    # 如果用户点击“结束游戏”
                    elif self.gameover_rect.left < pos[0] < self.gameover_rect.right and \
                            self.gameover_rect.top < pos[1] < self.gameover_rect.bottom:
                        # 退出游戏
                        pygame.quit()
                        sys.exit()
                        # 绘制暂停按钮

            self.screen.blit(self.paused_image, self.paused_rect)

            # 切换图片
            if not (self.delay % 5):
                self.switch_image = not self.switch_image

            self.delay -= 1
            if not self.delay:
                self.delay = 100

            pygame.display.flip()
            self.clock.tick(60)
            
    def train(self):
        
        QL = q_learning.approximateQlearning()
        for _ in range(3):
        
            # Initialize game
            self.init_game()

            newState = state.state(mePos=(self.me.rect.left, self.me.rect.top))
            
            while True:
                current_score = self.score
                currentState = newState
                action = QL.getAction(currentState)
                #action = random.choice(['left', 'right', 'up', 'down', 'bomb', 'stay'])
                
                self.print_info()
                
                self._handle_events()
                self._add_difficulty()

                if not self.paused and self.life_num:
                    
                    self.move_action(action)
                    
                    # 画面中的敌机位置
                    enemyPos = []
                    for enemy_1 in self.small_enemies.sprites():
                        if enemy_1.rect.top > 0:
                            enemyPos.append((enemy_1.rect.left, enemy_1.rect.top))
                    for enemy_2 in self.mid_enemies.sprites():
                        if enemy_2.rect.top > 0:
                            enemyPos.append((enemy_2.rect.left, enemy_2.rect.top))
                    for enemy_3 in self.big_enemies.sprites():
                        if enemy_3.rect.top > 0:
                            enemyPos.append((enemy_3.rect.left, enemy_3.rect.top))
                    
                    bullet_supply_pos = None
                    if self.bullet_supply.active:
                        if self.bullet_supply.rect.top > 0:
                            bullet_supply_pos = (self.bullet_supply.rect.left, self.bullet_supply.rect.top)
                    
                    bomb_supply_pos = None
                    if self.bomb_supply.active:
                        if self.bomb_supply.rect.top > 0:
                            bomb_supply_pos = (self.bomb_supply.rect.left, self.bomb_supply.rect.top)
                            
                    newState.assignData(
                        score=self.score,
                        mePos=(self.me.rect.left, self.me.rect.top),
                        enemy_num=len(enemyPos),
                        enemyPos=enemyPos,
                        life_num=self.life_num,
                        is_double_bullet=self.bullet_supply.active,
                        bomb_num=self.bomb_num,
                        bullet_supply_pos=bullet_supply_pos,
                        bomb_supply_pos=bomb_supply_pos,
                    )
                    
                    QL.update(currentState, action, newState, self.score - current_score)
                    
                    
                    self._draw_frame()
                elif self.life_num == 0:
                    # 结束游戏之后开始下一局训练
                    self._draw_game_over()
                    QL.update(currentState, action, newState, -10000)
                    # QL.StoreWeights()
                    pygame.display.flip()
                    print('Game ended. Wait 3s.')
                    time.sleep(3)
                    break

                self.screen.blit(self.paused_image, self.paused_rect)

                # 切换图片
                if not (self.delay % 5):
                    self.switch_image = not self.switch_image

                self.delay -= 1
                if not self.delay:
                    self.delay = 100

                pygame.display.flip()
                self.clock.tick(60)

    def print_info(self) -> None:
        """Print一些信息"""
        print_green("Me top={}, left={} ".format(self.me.rect.top, self.me.rect.left), end='')
        print_green("Bomb {}".format(self.bomb_num), end='')
        for enemy_1 in self.small_enemies.sprites():
            if enemy_1.rect.top > 0:
                print_red("small left={}, top={} ".format(enemy_1.rect.left, enemy_1.rect.top), end='')
        for enemy_1 in self.mid_enemies.sprites():
            if enemy_1.rect.top > 0:
                print_red("mid left={}, top={} ".format(enemy_1.rect.left, enemy_1.rect.top), end='')
        for enemy_1 in self.big_enemies.sprites():
            if enemy_1.rect.top > 0:
                print_red("big left={}, top={} ".format(enemy_1.rect.left, enemy_1.rect.top), end='')
        print()

    def _handle_events(self) -> None:
        """
        Handle interaction events.
        Checks whether game quit button (ESC) is pressed, game pause button is clicked,
        and checks if user has used a bomb.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and self.paused_rect.collidepoint(event.pos):
                    self.paused = not self.paused
                    if self.paused:
                        pygame.time.set_timer(self.SUPPLY_TIME, 0)
                        pygame.mixer.music.pause()
                        pygame.mixer.pause()
                    else:
                        pygame.time.set_timer(self.SUPPLY_TIME, 30 * 1000)
                        pygame.mixer.music.unpause()
                        pygame.mixer.unpause()
            elif event.type == pygame.MOUSEMOTION:
                if self.paused_rect.collidepoint(event.pos):
                    if self.paused:
                        self.pause_image = self.resume_pressed_image
                    else:
                        self.pause_image = self.pause_pressed_image
                else:
                    if self.paused:
                        self.pause_image = self.resume_nor_image
                    else:
                        self.pause_image = self.pause_nor_image
            # elif event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_SPACE:
            #         if self.bomb_num:
            #             self.bomb_num -= 1
            #             self.sounds['bomb'].play()
            #             for each in self.enemies:
            #                 if each.rect.bottom > 0:
            #                     each.active = False
            elif event.type == self.SUPPLY_TIME:
                self.sounds['supply'].play()
                if random.choice([True, False]):
                    self.bomb_supply.reset()
                else:
                    self.bullet_supply.reset()
            elif event.type == self.DOUBLE_BULLET_TIME:
                self.is_double_bullet = False
                pygame.time.set_timer(self.DOUBLE_BULLET_TIME, 0)
            elif event.type == self.INVINCIBLE_TIME:
                self.me.invincible = False
                pygame.time.set_timer(self.INVINCIBLE_TIME, 0)

    def _add_difficulty(self):
        """Increase difficulty according to current score. 根据当前得分增加难度"""

        if self.level == 1 and self.score > 50000:
            self.level = 2
            self.sounds['upgrade'].play()
            # 增加3架小型敌机、2架中型敌机和1架大型敌机
            self._add_small_enemies(3)
            self._add_mid_enemies(2)
            self._add_big_enemies(1)

        elif self.level == 2 and self.score > 300000:
            self.level = 3
            self.sounds['upgrade'].play()
            # 增加5架小型敌机、3架中型敌机和2架大型敌机
            self._add_small_enemies(5)
            self._add_mid_enemies(3)
            self._add_big_enemies(2)

        elif self.level == 3 and self.score > 600000:
            self.level = 4
            self.sounds['upgrade'].play()
            # 增加5架小型敌机、3架中型敌机和2架大型敌机
            self._add_small_enemies(5)
            self._add_mid_enemies(3)
            self._add_big_enemies(2)

    def move_me(self):
        """Move the plane by keyboard inputs. 根据用户键盘操作移动飞机"""
        key_pressed = pygame.key.get_pressed()
        if key_pressed[pygame.K_w] or key_pressed[pygame.K_UP]:
            self.me.moveUp()
        if key_pressed[pygame.K_s] or key_pressed[pygame.K_DOWN]:
            self.me.moveDown()
        if key_pressed[pygame.K_a] or key_pressed[pygame.K_LEFT]:
            self.me.moveLeft()
        if key_pressed[pygame.K_d] or key_pressed[pygame.K_RIGHT]:
            self.me.moveRight()

    def move_action(self, action: str):
        """Move the plane by the given action. 根据指定的操作移动飞机"""
        if action == 'stay':
            pass
        elif action == 'up':
            self.me.moveUp()
        elif action == 'down':
            self.me.moveDown()
        elif action == 'left':
            self.me.moveLeft()
        elif action == 'right':
            self.me.moveRight()
        elif action == 'bomb':
            if self.bomb_num:
                self.bomb_num -= 1
                self.sounds['bomb'].play()
                for each in self.enemies:
                    if each.rect.bottom > 0:
                        each.active = False
        else:
            raise KeyError('Unknown action type: {}'.format(action))

    def _draw_frame(self):
        """Calculate a frame (including positions and collisions) and render it. 计算物体位置(含碰撞检测)并渲染一帧"""

        global bullets
        
        # 清屏
        self.screen.blit(self.background, (0, 0))

        # 绘制全屏炸弹补给并检测是否获得
        if self.bomb_supply.active:
            self.bomb_supply.move()
            self.screen.blit(self.bomb_supply.image, self.bomb_supply.rect)
            if pygame.sprite.collide_mask(self.bomb_supply, self.me):
                self.sounds['get_bomb'].play()
                if self.bomb_num < 3:
                    self.bomb_num += 1
                self.bomb_supply.active = False

        # 绘制子弹补给并检测是否获得
        if self.bullet_supply.active:
            self.bullet_supply.move()
            self.screen.blit(self.bullet_supply.image, self.bullet_supply.rect)
            if pygame.sprite.collide_mask(self.bullet_supply, self.me):
                self.sounds['get_bullet'].play()
                # 发射超级子弹
                self.is_double_bullet = True
                pygame.time.set_timer(self.DOUBLE_BULLET_TIME, 18 * 1000)
                self.bullet_supply.active = False

        # 发射子弹
        if not (self.delay % 10):
            self.sounds['bullet'].play()
            if self.is_double_bullet:
                bullets = self.bullet_2
                bullets[self.bullet_2_index].reset((self.me.rect.centerx - 33, self.me.rect.centery))
                bullets[self.bullet_2_index + 1].reset((self.me.rect.centerx + 30, self.me.rect.centery))
                self.bullet_2_index = (self.bullet_2_index + 2) % 8
            else:
                bullets = self.bullet_1
                bullets[self.bullet_1_index].reset(self.me.rect.midtop)
                self.bullet_1_index = (self.bullet_1_index + 1) % 4

        # 检测子弹是否击中敌机
        for b in bullets:
            if b.active:
                b.move()
                self.screen.blit(b.image, b.rect)
                enemy_hit = pygame.sprite.spritecollide(b, self.enemies, False, pygame.sprite.collide_mask)
                if enemy_hit:  # 如果子弹打到了一架敌机
                    b.active = False
                    for e in enemy_hit:  # 如果打到了中型或大型敌机，则该敌机血量减1
                        if e in self.mid_enemies or e in self.big_enemies:
                            e.energy -= 1
                            if e.energy == 0:
                                e.active = False
                        else:
                            e.active = False

        # 绘制大型敌机
        for each in self.big_enemies:
            if each.active:
                each.move()
                if each.hit:
                    self.screen.blit(each.image_hit, each.rect)
                    each.hit = False
                else:
                    if self.switch_image:
                        self.screen.blit(each.image1, each.rect)
                    else:
                        self.screen.blit(each.image2, each.rect)
                # 绘制血槽
                pygame.draw.line(self.screen, (0, 0, 0),
                                 (each.rect.left, each.rect.top - 5), (each.rect.right, each.rect.top - 5), 2)
                energy_remain = each.energy / enemy.BigEnemy.energy
                if energy_remain > 0.2:
                    energy_color = (0, 255, 0)
                else:
                    energy_color = (255, 0, 0)
                pygame.draw.line(self.screen, energy_color,
                                 (each.rect.left, each.rect.top - 5),
                                 (each.rect.left + each.rect.width * energy_remain, each.rect.top - 5), 2)

                # 即将出现在画面中，播放音效
                if each.rect.bottom == -50:
                    self.sounds['enemy3_flying'].play(-1)

            else:  # 毁灭
                # 毁灭音效
                if not (self.delay % 3):
                    if self.e3_destroy_index == 0:
                        self.sounds['enemy3_down'].play()
                    self.screen.blit(each.destroy_images[self.e3_destroy_index], each.rect)
                    self.e3_destroy_index = (self.e3_destroy_index + 1) % 6
                    if self.e3_destroy_index == 0:
                        self.score += 10000
                        each.reset()
                        self.sounds['enemy3_flying'].stop()

        # 绘制中型敌机：
        for each in self.mid_enemies:
            if each.active:
                each.move()

                if each.hit:
                    self.screen.blit(each.image_hit, each.rect)
                    each.hit = False
                else:
                    self.screen.blit(each.image, each.rect)

                # 绘制血槽
                pygame.draw.line(self.screen, (0, 0, 0),
                                 (each.rect.left, each.rect.top - 5), (each.rect.right, each.rect.top - 5), 2)
                # 当生命大于20%显示绿色，否则显示红色
                energy_remain = each.energy / enemy.MidEnemy.energy
                if energy_remain > 0.2:
                    energy_color = (0, 255, 0)
                else:
                    energy_color = (255, 0, 0)
                pygame.draw.line(self.screen, energy_color,
                                 (each.rect.left, each.rect.top - 5),
                                 (each.rect.left + each.rect.width * energy_remain, each.rect.top - 5), 2)
            else:
                # 毁灭
                if not (self.delay % 3):
                    if self.e2_destroy_index == 0:
                        self.sounds['enemy2_down'].play()
                    self.screen.blit(each.destroy_images[self.e2_destroy_index], each.rect)
                    self.e2_destroy_index = (self.e2_destroy_index + 1) % 4
                    if self.e2_destroy_index == 0:
                        self.score += 6000
                        each.reset()

        # 绘制小型敌机
        for each in self.small_enemies:
            if each.active:
                each.move()
                self.screen.blit(each.image, each.rect)
            else:
                # 毁灭
                if not (self.delay % 3):
                    if self.e1_destroy_index == 0:
                        self.sounds['enemy1_down'].play()
                    self.screen.blit(each.destroy_images[self.e1_destroy_index], each.rect)
                    self.e1_destroy_index = (self.e1_destroy_index + 1) % 4
                    if self.e1_destroy_index == 0:
                        self.score += 1000
                        each.reset()

        # 检测我方飞机是否被撞
        enemies_down = pygame.sprite.spritecollide(self.me, self.enemies, False, pygame.sprite.collide_mask)
        if enemies_down and not self.me.invincible:
            self.me.active = False
            for e in enemies_down:
                e.active = False

        # 绘制我方飞机
        if self.me.active:
            if self.switch_image:
                self.screen.blit(self.me.image1, self.me.rect)
            else:
                self.screen.blit(self.me.image2, self.me.rect)
        else:
            if not (self.delay % 3):
                if self.me_destroy_index == 0:
                    self.sounds['me_down'].play()
                self.screen.blit(self.me.destroy_images[self.me_destroy_index], self.me.rect)
                self.me_destroy_index = (self.me_destroy_index + 1) % 4
                if self.me_destroy_index == 0:
                    self.life_num -= 1
                    self.me.reset()
                    pygame.time.set_timer(self.INVINCIBLE_TIME, 3 * 1000)

        # 绘制全屏炸弹数量
        bomb_text = self.bomb_font.render('x %d' % self.bomb_num, True, (255, 255, 255))
        text_rect = bomb_text.get_rect()
        self.screen.blit(self.bomb_image, (10, self.height - 10 - self.bomb_rect.height))
        self.screen.blit(bomb_text, (20 + self.bomb_rect.width, self.height - 5 - text_rect.height))

        # 绘制剩余生命数量
        if self.life_num:
            for i in range(self.life_num):
                self.screen.blit(self.life_image,
                                 (self.width - 10 - (i + 1) * self.life_rect.width,
                                  self.height - 10 - self.life_rect.height))

        # 绘制得分
        score_text = self.score_font.render('Score : %s' % str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (10, 5))

    def _draw_game_over(self) -> None:
        """
        When game over, this function is called to display the game over page and the score.
        """
        
        # 清屏
        self.screen.blit(self.background, (0, 0))
        
        # 停止播放音乐和音效, 停止计时器
        pygame.mixer.music.stop()
        pygame.mixer.stop()
        pygame.time.set_timer(self.SUPPLY_TIME, 0)

        record_score = 0
        if not self.recorded:
            self.recorded = True
    
            # 读取历史最高得分
            with open('record.txt', 'r') as f:
                record_score = int(f.read())
            # 如果玩家得分高于历史最高得分
            if self.score > record_score:
                with open('record.txt', 'w') as f:
                    f.write(str(self.score))

        record_score_text = self.score_font.render('Best : %d' % record_score, True, (255, 255, 255))
        self.screen.blit(record_score_text, (50, 50))

        # Render "Your score" with white color
        score_txt1 = self.gameover_font.render('Your Score', True, (255, 255, 255))
        score_rect1 = score_txt1.get_rect()
        score_rect1.left, score_rect1.top = (self.width - score_rect1.width) // 2, self.height // 3
        self.screen.blit(score_txt1, score_rect1)

        # Render score (a number) with white color
        score_txt2 = self.gameover_font.render(str(self.score), True, (255, 255, 255))
        score_rect2 = score_txt2.get_rect()
        score_rect2.left, score_rect2.top = (self.width - score_rect2.width) // 2, score_rect1.bottom + 10
        self.screen.blit(score_txt2, score_rect2)

        self.again_rect.left, self.again_rect.top = (self.width - self.again_rect.width) // 2, score_rect2.bottom + 50
        self.screen.blit(self.again_image, self.again_rect)

        self.gameover_rect.left, self.gameover_rect.top = (self.width - self.again_rect.width) // 2, self.again_rect.bottom + 10
        self.screen.blit(self.gameover_image, self.gameover_rect)

    def _add_small_enemies(self, num: int) -> None:
        """Add a given number of small-enemies to the game. 添加指定数量的小型敌机"""
        for _ in range(num):
            e1 = enemy.SmallEnemy((self.width, self.height))
            self.enemies.add(e1)
            self.small_enemies.add(e1)

    def _add_mid_enemies(self, num: int) -> None:
        """Add a given number of mid-enemies to the game. 添加指定数量的中型敌机"""
        for _ in range(num):
            e2 = enemy.MidEnemy((self.width, self.height))
            self.enemies.add(e2)
            self.mid_enemies.add(e2)

    def _add_big_enemies(self, num: int) -> None:
        """Add a given number of big-enemies to the game. 添加指定数量的大型敌机"""
        for _ in range(num):
            e3 = enemy.BigEnemy((self.width, self.height))
            self.enemies.add(e3)
            self.big_enemies.add(e3)


if __name__ == '__main__':
    try:
        game = PlaneWar()
        # game.start()  # 键盘控制
        game.train()  # Q-learning训练
        # 这个类
    except SystemExit:
        pass
    except:
        traceback.print_exc()
        pygame.quit()
