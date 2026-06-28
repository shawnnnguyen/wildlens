import { useEffect } from 'react';
import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { useFonts } from 'expo-font';
import { SpaceMono_400Regular, SpaceMono_700Bold } from '@expo-google-fonts/space-mono';
import { EBGaramond_400Regular, EBGaramond_400Regular_Italic } from '@expo-google-fonts/eb-garamond';
import { SessionProvider } from '../store/session';
import { StatusBar } from 'expo-status-bar';

SplashScreen.preventAutoHideAsync();

export default function RootLayout() {
  const [loaded] = useFonts({
    SpaceMono_400Regular,
    SpaceMono_700Bold,
    EBGaramond_400Regular,
    EBGaramond_400Regular_Italic,
    'CormorantGaramond-SemiBold':       require('../assets/fonts/CormorantGaramond-SemiBold.ttf'),
    'CormorantGaramond-Bold':           require('../assets/fonts/CormorantGaramond-Bold.ttf'),
    'CormorantGaramond-SemiBoldItalic': require('../assets/fonts/CormorantGaramond-SemiBoldItalic.ttf'),
  });

  useEffect(() => {
    if (loaded) SplashScreen.hideAsync();
  }, [loaded]);

  if (!loaded) return null;

  return (
    <SessionProvider>
      <StatusBar style="light" />
      <Stack screenOptions={{ headerShown: false, animation: 'fade' }}>
        <Stack.Screen name="index" />
        <Stack.Screen name="capture" />
        <Stack.Screen name="identified" />
        <Stack.Screen name="chat" />
        <Stack.Screen name="safety" options={{ animation: 'slide_from_bottom' }} />
        <Stack.Screen name="unclear" />
      </Stack>
    </SessionProvider>
  );
}
