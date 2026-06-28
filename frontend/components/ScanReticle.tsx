import { useEffect } from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, {
  useSharedValue, useAnimatedStyle,
  withRepeat, withSequence, withTiming, Easing,
} from 'react-native-reanimated';
import { Colors } from '../constants/theme';

const CORNER = 34;
const SIZE   = 212;
const BORDER = 2;

export default function ScanReticle() {
  const scale   = useSharedValue(1);
  const opacity = useSharedValue(0.85);

  useEffect(() => {
    scale.value = withRepeat(
      withSequence(
        withTiming(1.05, { duration: 1400, easing: Easing.inOut(Easing.ease) }),
        withTiming(1,    { duration: 1400, easing: Easing.inOut(Easing.ease) }),
      ),
      -1, false,
    );
    opacity.value = withRepeat(
      withSequence(
        withTiming(1,    { duration: 1400 }),
        withTiming(0.85, { duration: 1400 }),
      ),
      -1, false,
    );
  }, []);

  const animStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: opacity.value,
  }));

  const corner = (pos: object) => (
    <View style={[styles.corner, pos]} />
  );

  return (
    <Animated.View style={[styles.container, animStyle]}>
      {/* top-left */}
      <View style={[styles.corner, styles.tl, { borderTopWidth: BORDER, borderLeftWidth: BORDER, borderTopLeftRadius: 8 }]} />
      {/* top-right */}
      <View style={[styles.corner, styles.tr, { borderTopWidth: BORDER, borderRightWidth: BORDER, borderTopRightRadius: 8 }]} />
      {/* bottom-left */}
      <View style={[styles.corner, styles.bl, { borderBottomWidth: BORDER, borderLeftWidth: BORDER, borderBottomLeftRadius: 8 }]} />
      {/* bottom-right */}
      <View style={[styles.corner, styles.br, { borderBottomWidth: BORDER, borderRightWidth: BORDER, borderBottomRightRadius: 8 }]} />
      <View style={styles.dot} />
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  container: { width: SIZE, height: SIZE },
  corner: {
    position: 'absolute',
    width: CORNER, height: CORNER,
    borderColor: 'rgba(243,236,222,0.92)',
  },
  tl: { top: 0, left: 0 },
  tr: { top: 0, right: 0 },
  bl: { bottom: 0, left: 0 },
  br: { bottom: 0, right: 0 },
  dot: {
    position: 'absolute',
    top: SIZE / 2 - 3, left: SIZE / 2 - 3,
    width: 6, height: 6, borderRadius: 3,
    backgroundColor: Colors.amber,
  },
});
