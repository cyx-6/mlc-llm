//
//  LLMChat.mm
//  LLMChat
//
#import <Foundation/Foundation.h>
#include <os/proc.h>
#include <iostream>

#include "LLMChat.h"

@implementation ChatModule {
    // Internal c++ classes
    std::string text;
    int count;
}

- (instancetype)init {
    text = "";
    count = 0;
    return self;
}

- (void)unload {
    text = "";
    count = 0;
    sleep(1);
}

- (void)reload:(NSString*)modelLib modelPath:(NSString*)modelPath {
    text = "";
    count = 0;
    sleep(1);
}

- (void)resetChat {
    text = "";
    count = 0;
    sleep(1);
}

- (void)prefill:(NSString*)input {
    text = input.UTF8String;
    count = 0;
    sleep(1);
}

- (void)decode {
    ++count;
    sleep(1);
}

- (NSString*)getMessage {
    return [@"" stringByPaddingToLength: count*(text.length()) withString: [NSString stringWithUTF8String:text.c_str()] startingAtIndex:0];
}

- (bool)stopped {
    return count >= 10;
}

- (NSString*)runtimeStatsText {
    return [NSString stringWithUTF8String:text.c_str()];
}

- (void)evaluate {
    
}

@end
