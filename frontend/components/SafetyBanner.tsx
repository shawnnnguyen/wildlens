import { useEffect } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue, useAnimatedStyle,
  withRepeat, withSequence, withTiming,
} from 'react-native-reanimated';
import { Colors, Fonts } from '../constants/theme';

interface Props { label: string }

export default function SafetyBanner({ label }: Props) {
  const opacity = useSharedValue(1);

  useEffect(() => {
    opacity.value = withRepeat(
      withSequence(
        withTiming(0.2, { duration: 500 }),
        withTiming(1,   { duration: 500 }),
      ),
      -1, false,
    );
  }, []);

  const dotStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

  return (
    <View style={styles.banner}>
      <Animated.View style={[styles.dot, dotStyle]} />
      <Text style={styles.text}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  banner: {
    flexDirection: 'row', alignItems: 'center', gap: 11,
    backgroundColor: Colors.danger,
    marginHorizontal: 16, marginTop: 6,
    padding: 13, paddingHorizontal: 16,
    borderRadius: 8,
  },
  dot: { width: 9, height: 9, borderRadius: 4.5, backgroundColor: '#fff' },
  text: { fontFamily: Fonts.mono, fontSize: 11, letterSpacing: 1.4, color: '#fff' },
});
